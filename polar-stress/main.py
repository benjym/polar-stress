import sys, os
import json5
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import image
import generate
import plotting

# from plotting import *
from tqdm import tqdm

params = json5.load(open("json/EVT.json5", "r"))

# Load the data
data = tifffile.imread(params["filename"])
data = image.gray_to_rgb(data)
data = generate.add_uniform_polarisation_to_image(data, params["polarisation"])

# Calculate the Degree of Linear Polarisation (DoLP)
DoLP = image.DoLP(data)
AoLP = image.AoLP(data)

# Plot the results
plotting.plot_DoLP_AoLP(DoLP, AoLP, filename="output.png")
