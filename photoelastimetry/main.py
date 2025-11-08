import os
import argparse
import json5
import numpy as np
from scipy.ndimage import gaussian_filter
import photoelastimetry.plotting
import photoelastimetry.local
import photoelastimetry.io


def image_to_stress(params, output_filename=None, polariser_angle_deg=0.0):
    """
    Convert photoelastic images to stress maps.

    This function processes raw photoelastic data to recover stress distribution maps
    using the stress-optic law and polarization analysis.

    Args:
        params (dict): Configuration dictionary containing:
            - folderName (str): Path to folder containing raw photoelastic images
            - crop (list, optional): Crop region as [y1, y2, x1, x2]
            - debug (bool): If True, display all channels for debugging
            - C (float): Stress-optic coefficient in 1/Pa
            - thickness (float): Sample thickness in meters
            - wavelengths (list): List of wavelengths in nanometers
        output_filename (str, optional): Path to save the output stress map image.
            If None, the stress map is not saved. Defaults to None.
        polariser_angle_deg (float, optional): Polariser angle in degrees relative to the 0 degree camera axis.
            Defaults to 0.0.

    Returns:
        numpy.ndarray: 2D array representing the stress map in Pascals.

    Notes:
        - Assumes incoming light is fully S1 polarized
        - Uses uniform stress-optic coefficient across all wavelengths
        - Assumes solid sample (NU = 1.0)
        - Wavelengths are automatically converted from nm to meters
    """

    data, metadata = photoelastimetry.io.load_raw(params["folderName"])

    if params.get("crop") is not None:
        data = data[
            params["crop"][0] : params["crop"][1],
            params["crop"][2] : params["crop"][3],
            :,
            :,
        ]

    if params["debug"]:
        photoelastimetry.plotting.show_all_channels(data, metadata)

    C = params["C"]  # Stress-optic coefficients in 1/Pa
    L = params["thickness"]  # Thickness in m
    wavelengths_nm = np.array(params["wavelengths"])  # Wavelengths in nm
    NU = 1.0  # Solid sample
    WAVELENGTHS = wavelengths_nm * 1e-9  # Convert to meters
    if isinstance(C, list) or isinstance(C, np.ndarray):
        C_VALUES = C
    else:
        C_VALUES = [
            C,
            C,
            C,
        ]  # Stress-optic coefficients in 1/Pa

    polariser_angle_rad = np.deg2rad(polariser_angle_deg)
    # Assume light fully polarised in polariser direction
    S_I_HAT = np.array([np.cos(polariser_angle_rad), np.sin(polariser_angle_rad)])
    # S_I_HAT = np.array([1.0, 0.0])  # Incoming light is fully S1 polarized

    # Calculate stress map from image
    stress_map = photoelastimetry.local.recover_stress_map(
        data,
        WAVELENGTHS,
        C_VALUES,
        NU,
        L,
        S_I_HAT,
    )
    if output_filename is not None:
        photoelastimetry.io.save_image(output_filename, stress_map, metadata)


def stress_to_image(params):
    """
    Convert stress field data to photoelastic fringe pattern image.

    This function loads stress field data, optionally applies Gaussian scattering,
    computes principal stresses and their orientations, calculates photoelastic
    retardation and fringe patterns, and saves the resulting visualization.

    Args:
        params (dict): Dictionary containing the following keys:
            - p_filename (str): Path to the photoelastimetry parameter file
            - stress_filename (str): Path to the stress field data file
            - scattering (float, optional): Gaussian filter sigma for scattering simulation.
              If falsy, no scattering is applied.
            - t (float): Thickness of the photoelastic material
            - lambda_light (float): Wavelength of light used in the experiment
            - C (float): Stress-optic coefficient of the material
            - output_filename (str, optional): Path for the output image.
              Defaults to "output.png" if not provided.

    Returns:
        None: The function saves the fringe pattern visualization to a file.

    Notes:
        - The stress field is expected to have components in the order [sigma_xy, sigma_yy, sigma_xx]
        - Principal stresses are computed using Mohr's circle equations
        - Isochromatic fringe intensity is calculated using sin²(δ/2)
        - Isoclinic angle represents the orientation of principal stresses
    """

    with open(params["p_filename"]) as f:
        dict, p = photoelastimetry.io.load_file(f)

    sigma = photoelastimetry.io.load_image(params["stress_filename"], dict)

    sigma_xx = sigma[:, :, 2]
    sigma_xy = sigma[:, :, 0]
    sigma_yy = sigma[:, :, 1]

    if params["scattering"]:
        # Add scattering
        sigma_xx = gaussian_filter(sigma_xx, sigma=params["scattering"])
        sigma_xy = gaussian_filter(sigma_xy, sigma=params["scattering"])
        sigma_yy = gaussian_filter(sigma_yy, sigma=params["scattering"])

    # Compute principal stresses
    sigma_avg = (sigma_xx + sigma_yy) / 2
    R = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + sigma_xy**2)
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R

    # Stress difference and retardation
    delta_sigma = sigma_1 - sigma_2

    # Retardation
    delta = (2 * np.pi * params["t"] / params["lambda_light"]) * params["C"] * delta_sigma

    # Fringe order
    # N = delta / (2 * np.pi)

    # Visualize Isochromatic Fringe Pattern
    fringe_intensity = np.sin(delta / 2) ** 2  # Fringe pattern

    # Isoclinic angle (principal stress orientation)
    phi = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)  # Angle in radians

    # Plot the results
    if "output_filename" in params:
        output_filename = params["output_filename"]
    else:
        output_filename = "output.png"
    photoelastimetry.plotting.plot_fringe_pattern(fringe_intensity, phi, filename=output_filename)


def cli_image_to_stress():
    """Command line interface for image_to_stress function."""
    parser = argparse.ArgumentParser(description="Convert photoelastic images to stress maps.")
    parser.add_argument(
        "json_filename",
        type=str,
        help="Path to the JSON5 parameter file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output stress map image (optional).",
    )
    parser.add_argument(
        "--polariser-angle",
        type=float,
        default=0.0,
        help="Polariser angle in degrees (default: 0.0).",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))
    image_to_stress(params, output_filename=args.output, polariser_angle_deg=args.polariser_angle)


def cli_stress_to_image():
    """Command line interface for stress_to_image function."""
    parser = argparse.ArgumentParser(
        description="Convert stress field data to photoelastic fringe pattern image."
    )
    parser.add_argument(
        "json_filename",
        type=str,
        help="Path to the JSON5 parameter file.",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))
    stress_to_image(params)


def demosaic_raw_image(input_file, metadata, output_prefix=None, output_format="tiff"):
    """
    De-mosaic a raw polarimetric image and save to TIFF stack or individual PNGs.

    This function takes a raw image from a polarimetric camera with a 4x4 superpixel
    pattern and splits it into separate channels for each color and polarization angle.

    Args:
        input_file (str): Path to the raw image file.
        metadata (dict): Dictionary containing image metadata with keys:
            - width (int): Image width in pixels
            - height (int): Image height in pixels
            - dtype (str, optional): Data type ('uint8' or 'uint16')
        output_prefix (str, optional): Prefix for output files. If None, uses input
            filename without extension. Defaults to None.
        output_format (str, optional): Output format, either 'tiff' for a single
            TIFF stack or 'png' for individual PNG files. Defaults to 'tiff'.

    Returns:
        numpy.ndarray: De-mosaiced image stack of shape [H, W, 4, 4] where:
            - H, W are the de-mosaiced dimensions (1/4 of original)
            - First dimension 4: color channels (R, G1, G2, B)
            - Second dimension 4: polarization angles (0°, 45°, 90°, 135°)

    Notes:
        - The raw image uses a 4x4 superpixel pattern with interleaved polarization
          and color filters
        - Output TIFF stack has shape [H, W, 4, 4] with all channels
        - Output PNGs create 4 files (one per polarization angle) with shape [H, W, 4]
          showing all color channels
    """
    # Read raw image
    data = photoelastimetry.io.read_raw(input_file, metadata)

    # De-mosaic into channels
    demosaiced = photoelastimetry.io.split_channels(data)

    # Keep only R, G1, B channels by removing G2
    demosaiced = demosaiced[:, :, [0, 1, 3], :]  # Keep R, G1, B

    # Determine output filename prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(input_file)[0]

    # Save based on format
    if output_format.lower() == "tiff":
        import tifffile

        output_file = f"{output_prefix}_demosaiced.tiff"
        # Permute to [4, 3, H, W] so TIFF is interpreted as 4 timepoints of 3-channel images
        demosaiced_transposed = np.transpose(demosaiced, (3, 2, 0, 1))
        tifffile.imwrite(
            output_file, demosaiced_transposed.astype(np.uint16), imagej=True, metadata={"axes": "TCYX"}
        )
        print(f"Saved TIFF stack to {output_file} (4 polarization angles × 3 color channels)")
    elif output_format.lower() == "png":
        import matplotlib.pyplot as plt

        angle_names = ["0deg", "45deg", "90deg", "135deg"]
        for i, angle in enumerate(angle_names):
            output_file = f"{output_prefix}_{angle}.png"

            # Normalize to 0-1 for PNG
            # HARDCODED: 4096 for 12-bit images
            img = demosaiced[:, :, :, i] / 4096

            plt.imsave(output_file, img)
            print(f"Saved {angle} polarization to {output_file}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Use 'tiff' or 'png'.")

    return demosaiced


def cli_demosaic():
    """Command line interface for de-mosaicing raw polarimetric images."""
    parser = argparse.ArgumentParser(description="De-mosaic raw polarimetric images into separate channels.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the raw image file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Image height in pixels.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["uint8", "uint16"],
        help="Data type (uint8 or uint16). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: input filename without extension).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="tiff",
        choices=["tiff", "png"],
        help="Output format: 'tiff' for single stack, 'png' for 4 separate images (default: tiff).",
    )
    args = parser.parse_args()

    metadata = {
        "width": args.width,
        "height": args.height,
    }
    if args.dtype:
        metadata["dtype"] = args.dtype

    demosaic_raw_image(args.input_file, metadata, args.output_prefix, args.format)
