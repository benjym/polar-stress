import sys, os
import argparse
import json5
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import polar_stress.plotting
import polar_stress.image
import polar_stress.io


def image_to_stress(params):
    data, metadata = polar_stress.io.load_raw(params["folderName"])
    if params.get("crop") is not None:
        data = data[
            params["crop"][0] : params["crop"][1],
            params["crop"][2] : params["crop"][3],
            :,
            :,
        ]

    if params["debug"]:
        polar_stress.plotting.show_all_channels(data, metadata)

    # Calculate the Degree of Linear Polarisation (DoLP)
    DoLP = polar_stress.image.DoLP(data)
    AoLP = polar_stress.image.AoLP(data)

    # Plot the results
    polar_stress.plotting.plot_DoLP_AoLP(DoLP, AoLP, filename="DoLP_AoLP.png")


def stress_to_image(params):
    from HGD.params import load_file
    from HGD.stress import calculate_stress
    from HGD.operators import get_solid_fraction

    with open(params["p_filename"]) as f:
        dict, p = load_file(f)
    p.update_before_time_march(None)
    s = np.load(params["s_filename"])
    nu = get_solid_fraction(s)
    sigma = calculate_stress(s, None, p)

    sigma_xx = sigma[:, :, 2]
    sigma_xy = sigma[:, :, 0]
    sigma_yy = sigma[:, :, 1]

    if params["scattering"]:
        # Add scattering
        sigma_xx = gaussian_filter(sigma_xx, sigma=params["scattering"])
        sigma_xy = gaussian_filter(sigma_xy, sigma=params["scattering"])
        sigma_yy = gaussian_filter(sigma_yy, sigma=params["scattering"])

    # Example stress state arrays (2D grid)
    # sigma_xx = np.random.uniform(-10e6, 10e6, (100, 100))  # Pa
    # sigma_yy = np.random.uniform(-10e6, 10e6, (100, 100))  # Pa
    # tau_xy = np.random.uniform(-5e6, 5e6, (100, 100))  # Pa

    # Compute principal stresses
    sigma_avg = (sigma_xx + sigma_yy) / 2
    R = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + sigma_xy**2)
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R

    # Stress difference and retardation
    delta_sigma = sigma_1 - sigma_2

    delta = (
        (2 * np.pi * params["t"] / params["lambda_light"])
        * params["C"]
        * delta_sigma
    )  # Retardation

    # Fringe order
    N = delta / (2 * np.pi)

    # Visualize Isochromatic Fringe Pattern
    fringe_intensity = np.sin(delta / 2) ** 2  # Fringe pattern

    # Isoclinic angle (principal stress orientation)
    phi = 0.5 * np.arctan2(
        2 * sigma_xy, sigma_xx - sigma_yy
    )  # Angle in radians

    # Plot the results
    plotting.plot_fringe_pattern(fringe_intensity, phi, filename="output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process stress data and generate fringe patterns."
    )
    parser.add_argument(
        "json_filename",
        type=str,
        help="Path to the JSON file with parameters.",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))

    if params["mode"] == "camera":
        image_to_stress(params)
    elif params["mode"] == "HGD":
        stress_to_image(params)
