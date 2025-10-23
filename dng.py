#! /usr/bin/env python3

import rawpy
import numpy as np
import os
from pathlib import Path
import cv2
import argparse
import exifread
import sys
from network import Network
from tuning import Tuning
from rgb import save as save_rgb

class Dng:
    """
    A class to represent a DNG file.

    Defective pixel correction (DPC), lens shading correction (LSC), digital gain and denoise,
    using the supplied NAFNET models, can be applied directly to the raw image.

    The raw image can be saved to a new DNG file, and/or converted to an RGB image.
    """

    def __init__(self, dng_filename: str, tuning = None, sensor = None) -> None:
        """
        Initialize Dng object by loading a DNG file. Optionally, a Tuning object can be provided, otherwise it
        will be loaded from the DNG filename.

        Args:
            filename (str): Path to the DNG file
            tuning (Tuning): Tuning object for this camera
            sensor (str): Sensor model name, if not provided, it will be read from the DNG file
        """
        self.dng_filename = Path(dng_filename)
        if not self.dng_filename.exists():
            raise FileNotFoundError(f"DNG file {dng_filename} not found")

        self.raw = rawpy.imread(str(self.dng_filename))

        # rawpy doesn't read all the exif tags, so read the missing ones with exifread.
        with open(self.dng_filename, 'rb') as f:
            self.exif_data = exifread.process_file(f, details=True)
        self.model = sensor if sensor else self.exif_data["Image Model"].values
        # Some Picamera2 DNG files won't have a valid sensor recorded.
        if not sensor and self.model == "PiDNG / PiCamera2":
            raise ValueError("Sensor model not found in the DNG file - try the -s option to specify it")

        # Each pair holds the (x, y) offsets for the R, Gr, B and Gb channels respectively
        self.raw_offsets = [None, None, None, None]
        self.raw_offsets[self.raw.raw_pattern[0, 0]] = (0, 0)
        self.raw_offsets[self.raw.raw_pattern[0, 1]] = (0, 1)
        self.raw_offsets[self.raw.raw_pattern[1, 0]] = (1, 0)
        self.raw_offsets[self.raw.raw_pattern[1, 1]] = (1, 1)

        if tuning:
            self.tuning = tuning
        else:
            tuning_file = Tuning.find(self.model)
            self.tuning = Tuning.load(tuning_file)

    @property
    def raw_array(self) -> np.ndarray:
        """
        Return the raw Bayer image.
        """
        return self.raw.raw_image_visible

    @property
    def black_level(self) -> int:
        """
        Return the black level for the image (assume same for all channels).
        """
        return self.raw.black_level_per_channel[0]

    @property
    def white_level(self) -> int:
        """
        Return the white level for the image.
        """
        return self.raw.white_level

    @property
    def camera_white_balance(self) -> np.ndarray:
        """
        Return the camera white balance for the image.
        """
        return self.raw.camera_whitebalance

    def save(self, output_filename: str, overwrite: bool = False):
        """
        Save this DNG to a another DNG file. Does it by copying the original file and writing the new
        version of the raw data. Though perhaps a bit hacky, it should preserve all the other data in
        the original file correctly.

        Currently limited to DNG files where the raw image is stored in a single strip.

        Args:
            output_filename (str): Path to the output DNG file
            overwrite (bool): Whether to overwrite the output file if it already exists
        """
        if not overwrite and os.path.exists(output_filename):
            raise FileExistsError(f"File {output_filename} already exists")

        # Start by getting the original file's strip offset. We may be dealing with SubImage1
        # (files from rpicam-still) or just the Image (Picamera2).
        try:
            start_offset = self.exif_data["EXIF SubIFD0 StripOffsets"].values[0]
            length = self.exif_data["EXIF SubIFD0 StripByteCounts"].values[0]
        except KeyError:
            start_offset = self.exif_data["Image StripOffsets"].values[0]
            length = self.exif_data["Image StripByteCounts"].values[0]
        if self.raw_array.nbytes > length:
            raise ValueError("Internal image size error, or maybe a multi-strip file?")

        # Now copy the original file and write the new version of the raw data.
        with open(self.dng_filename, "rb") as in_file, open(output_filename, "wb") as out_file:
            preamble = in_file.read(start_offset)
            out_file.write(preamble)
            out_file.write(self.raw_array.tobytes())
            in_file.seek(self.raw_array.nbytes, os.SEEK_CUR)
            postamble = in_file.read()
            out_file.write(postamble)

    def close(self):
        """
        Close the DNG file to free resources.
        """
        self.raw.close()
        self.raw = None
        self.exif_data = None
        self.raw_offsets = None
        self.tuning = None
        self.model = None
        self.dng_filename = None

    def __del__(self):
        self.close()

    def do_dpc(self, extra: float = 0.25) -> None:
        """
        Apply simple DPC (Defective Pixel Correction) to the raw image. This alters the raw image in place,
        so it should not really be called more than once. It should work adequately for single pixel defects.

        Args:
            extra (float): Allow slightly wider limits for pixel clipping (default is 0.25)
        """
        # We're going to ignore the two edge rows/columns, unless we see a need later.
        arrays = [
            self.raw_array[:-4, :-4],
            self.raw_array[:-4, 2:-2],
            self.raw_array[:-4, 4:],
            self.raw_array[2:-2, :-4],
            self.raw_array[2:-2, 4:],
            self.raw_array[4:, :-4],
            self.raw_array[4:, 2:-2],
            self.raw_array[4:, 4:],
        ]
        max_array = np.max(arrays, axis=0)
        min_array = np.min(arrays, axis=0)
        centre = self.raw_array[2:-2, 2:-2]
        # Clip central pixel to the min/max of the neighbours, plus a little "extra".
        max_array = max_array.astype(np.float32)
        min_array = min_array.astype(np.float32)
        diff = (max_array - min_array) * extra
        max_array += diff
        min_array -= diff
        centre = centre.astype(np.float32).clip(min_array, max_array)

        self.raw_array[2:-2, 2:-2] = centre.clip(0, self.white_level).astype(np.uint16)

    def estimate_colour_temp(self) -> float:
        """
        Estimate the colour temperature of the image, using the camera white balance and the tuning file.

        Returns:
            float: Estimated colour temperature in Kelvin
        """
        red_blue = 1.0 / np.array(self.camera_white_balance)[[0, 2]]
        colour_temp = self.tuning.get_colour_temp(red_blue)
        return colour_temp

    def do_lsc(self, colour_temp: float = None) -> None:
        """
        Apply LSC (Lens Shading Correction) to the raw image. This alters the raw image in place,
        so it should not really be called more than once.

        Args:
            colour_temp (float): Colour temperature to use for LSC. If None, it will be estimated
                from the camera white balance and the tuning file.
        """
        raw_image = self.raw_array.astype(np.float32)

        # Subtract the black level.
        raw_image -= self.black_level
        raw_image = raw_image.clip(0, self.white_level)

        # Get the lens shading correction tables. First, we need to estimate the colour temperature.
        if colour_temp is None:
            colour_temp = self.estimate_colour_temp()
        r_table, g_table, b_table = self.tuning.get_lsc_tables(colour_temp)

        # Apply the lens shading correction.
        w, h = raw_image.shape[1::-1]
        half_res = (w // 2, h // 2)
        lsc_tables = [r_table, g_table, b_table, g_table]
        for component in range(4):
            offsets = self.raw_offsets[component][0], self.raw_offsets[component][1]
            raw_image[offsets[0]::2, offsets[1]::2] *= cv2.resize(lsc_tables[component], half_res)

        self.raw_array[...] = (raw_image + self.black_level).clip(0, self.white_level).astype(np.uint16)

    def do_digital_gain(self, digital_gain: float) -> None:
        """
        Apply digital gain to the raw image. This alters the raw image in place,
        so it should not really be called more than once.
        """
        array = self.raw_array.astype(np.float32) - self.black_level
        array *= digital_gain
        array += self.black_level
        self.raw_array[...] = array.clip(0, self.white_level).astype(np.uint16)

    def convert(self, colour_gains=None, gamma=None, median_filter_passes=1, output_bps=8) -> np.ndarray:
        """
        Convert the raw image to an RGB image using rawpy. You should consider whether you want
        to apply denoise, DPC or LSC before calling this.

        Args:
            colour_gains (tuple): If None, the camera white balance will be used. Otherwise, pass a pair of
                numbers defining the red and blue gains.
            gamma (tuple): If None, the gamma curve from the tuning file will be used. Otherwise, pass a pair of
                numbers defining a gamma curve in the manner or rawpy.
            median_filter_passes (int): Number of median filter passes to apply.
            output_bps (int): Output bit depth (8 or 16 bits only).

        Returns:
            np.ndarray: RGB image
        """
        use_camera_wb = True
        user_wb = None
        if colour_gains:
            use_camera_wb = False
            red, blue = colour_gains
            user_wb = [red, 1.0, blue, 1.0]
            min_gain = min(red, blue)
            user_wb = (np.array(user_wb) / min_gain).tolist()

        rgb_image = self.raw.postprocess(
            use_camera_wb=use_camera_wb,
            user_wb=user_wb,
            no_auto_bright=True,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DCB,
            median_filter_passes=median_filter_passes,
            gamma=(1.0, 1.0) if gamma is None else gamma,
            output_bps=16)

        # If no gamma was supplied, use the one from the tuning file.
        if gamma is None:
            gamma_curve = self.tuning.get_gamma_curve()
            gamma_lut = np.interp(range(65536), gamma_curve[0], gamma_curve[1], right=65535).astype(np.uint16)
            rgb_image[...] = gamma_lut[rgb_image]

        if output_bps == 16:
            pass  # should be 16 bit already
        elif output_bps == 8:
            rgb_image = (rgb_image >> 8).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported output bit depth: {output_bps}")
        return rgb_image

    def _make_BGGR(self, array: np.ndarray) -> np.ndarray:
        """
        Convert the raw array to BGGR order by performing horizontal and vertical flips.

        Also converts the BGGR array back to native Bayer order; the transform is the same.

        Returns:
            np.ndarray: Array of same shape, but (possibly flipped to be) in BGGR order
        """
        # 2 means "blue"
        if self.raw.raw_pattern[0, 0] == 2:
            return array  # already BGGR
        elif self.raw.raw_pattern[0, 1] == 2:
            return array[::, ::-1]  # need horizontal flip
        elif self.raw.raw_pattern[1, 0] == 2:
            return array[::-1, ::]  # need vertical flip
        elif self.raw.raw_pattern[1, 1] == 2:
            return array[::-1, ::-1]  # need vertical and horizontal flip
        else:
            raise ValueError("Invalid raw pattern")

    def _get_bayer_planes(self) -> np.ndarray:
        """
        Extract the four Bayer planes from the raw array in the order B, Gb, Gr, R.

        Returns:
            np.ndarray: Array of shape (height//2, width//2, 4) containing the four Bayer planes
        """
        # Get the raw array dimensions
        height, width = self.raw_array.shape

        # Create output array for the four planes
        planes = np.zeros((height//2, width//2, 4), dtype=np.float32)

        array_bggr = self._make_BGGR(self.raw_array)

        # Extract each plane, we can now assume BGGR order
        planes[..., 0] = array_bggr[0::2, 0::2]  # B
        planes[..., 1] = array_bggr[0::2, 1::2]  # Gb
        planes[..., 2] = array_bggr[1::2, 0::2]  # Gr
        planes[..., 3] = array_bggr[1::2, 1::2]  # R

        # Normalise the planes.
        planes -= self.black_level
        planes /= (self.white_level - self.black_level)
        planes[..., 0] *= max(self.camera_white_balance[2], 1)  # blue
        planes[..., 3] *= max(self.camera_white_balance[0], 1)  # red

        return planes

    def _set_bayer_planes(self, planes: np.ndarray) -> None:
        """
        Copy the four Bayer planes back into the raw array.

        Args:
            planes (np.ndarray): Array of shape (height//2, width//2, 4) containing the four Bayer planes
                                in the order B, Gb, Gr, R
        """
        # Get the raw array dimensions
        height, width = self.raw_array.shape

        # Verify input shape
        if planes.shape != (height//2, width//2, 4):
            raise ValueError(f"Expected planes shape {(height//2, width//2, 4)}, got {planes.shape}")

        planes[..., 0] /= max(self.camera_white_balance[2], 1)  # blue
        planes[..., 3] /= max(self.camera_white_balance[0], 1)  # red
        planes *= (self.white_level - self.black_level)
        planes += self.black_level + 0.5
        planes = planes.clip(0, self.white_level)

        # Turn the planes into a single BGGR array.
        array_bggr = np.empty((height, width), dtype=self.raw_array.dtype)
        array_bggr[0::2, 0::2] = planes[..., 0]  # B
        array_bggr[0::2, 1::2] = planes[..., 1]  # Gb
        array_bggr[1::2, 0::2] = planes[..., 2]  # Gr
        array_bggr[1::2, 1::2] = planes[..., 3]  # R

        # Convert the BGGR array back to native Bayer order before writing back to the raw array.
        self.raw_array[...] = self._make_BGGR(array_bggr)

    def do_denoise(self, network: Network, overlap_pixels: int = 16, show_progress: bool = True) -> None:
        """
        Apply denoising to the raw image. This alters the raw image in place,
        so it should not really be called more than once.
        """
        planes = self._get_bayer_planes()
        denoised_planes = network.run_inference(planes, overlap_pixels, show_progress)
        self._set_bayer_planes(denoised_planes)

# Helper function to parse string "num1,num2" to tuple of floats
def parse_two_values(num_str: str) -> tuple[float, float]:
    try:
        parts = num_str.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Must be two comma-separated numbers (e.g. '2.2,4.5')")
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError("Values must be numbers (e.g. '2.2,4.5')")

if __name__ == "__main__":
    """
    Command line interface to process a DNG file. The tool can be used to:

    * Apply any or all of DPC, LSC, digital gain, denoise (using a neural network model)
      to the raw image data, which can be saved to a new DNG file.
    * Additionaly, the raw data, after any processing, can be converted to an RGB image and
      saved to another file.

    Usage:

    python dng.py --input input.dng --output-rgb output.jpg
      Simple example where input.dng is converted to an RGB image and saved to output.jpg.
      LSC is applied by default, but there is no DPC, denoise or digital gain.

    python dng.py --input input.dng --output-dng output.dng --denoise on
      Apply denoise to the raw image data and save the result to output.dng.

    python dng.py --input input.dng --output-dng output.dng --denoise on --dpc on --network networks/nafnet_bayer_large.tflite
      Apply denoise and DPC to the raw image data and save the result to output.dng. Use the larger
      network model for better denoising.

    Notes:
    * When creating an RGB output, the --colour-gains and --gamma options can be used to adjust the
      output image.
    * At the time of writing, DNG files from Picamera2 don't record the sensor model, so the -s option
      should be used to specify it. The problem is being fixed in Picamera2 and PiDNG.

    Type "python dng.py --help" for more options.
    """

    default_network_path = str(Path(__file__).resolve().parent / "networks" / "nafnet_bayer_small.tflite")

    parser = argparse.ArgumentParser(description="Process a DNG file.")
    parser.add_argument("-i", "--input", required=True, help="Input DNG filename")
    parser.add_argument("-s", "--sensor", help="Sensor model name (e.g. imx477, optional though some DNG files may need it)")
    parser.add_argument("--overlap", type=int, default=16, help="Number of overlap pixels between image patches")
    parser.add_argument("--tuning", help="Tuning filename (optional)")
    parser.add_argument("-n", "--network", default=default_network_path, help=f"Network model filename (default: {str(default_network_path)})")
    parser.add_argument("--denoise", choices=["on", "off"], default="off", help="Enable or disable denoising (default: off)")
    parser.add_argument("--dpc", choices=["on", "off"], default="off", help="Enable or disable DPC (Defective Pixel Correction) (default: off)")
    parser.add_argument("--lsc", choices=["on", "off"], default="on", help="Enable or disable LSC (Lens Shading Correction) (default: on)")
    parser.add_argument("--colour-gains", type=parse_two_values, default=None, help="Red and blue gains as 'num1,num2' (e.g. '1.7,2.3'). Defaults to DNG file gains.")
    parser.add_argument("--digital-gain", type=float, default=1.0, help="Apply digital gain (default: 1.0)")
    parser.add_argument("--gamma", type=parse_two_values, default=None, help="Gamma curve as 'num1,num2' (e.g. '2.2,4.5') as per rawpy. Defaults to tuning file gamma.")
    parser.add_argument("--output-dng", help="Output DNG filename (optional)")
    parser.add_argument("--output-rgb", help="Output RGB filename (optional)")
    parser.add_argument("-y", "--yes", action="store_true", help="Overwrite existing output files (default: False)")
    args = parser.parse_args()

    tuning = None
    if args.tuning:
        tuning = Tuning.load(args.tuning)

    dng = Dng(args.input, sensor=args.sensor, tuning=tuning)

    if args.dpc == "on":
        dng.do_dpc()

    if args.denoise == "on":
        network = Network(args.network)
        dng.do_denoise(network, args.overlap)

    if args.lsc == "on":
        dng.do_lsc()

    if args.digital_gain != 1.0:
        dng.do_digital_gain(args.digital_gain)

    if args.output_dng:
        dng.save(args.output_dng, overwrite=args.yes)

    if args.output_rgb:
        rgb = dng.convert(colour_gains=args.colour_gains, gamma=args.gamma, median_filter_passes=1, output_bps=8)
        save_rgb(args.output_rgb, rgb, overwrite=args.yes)
