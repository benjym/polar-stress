import sys, os
import json5
import numpy as np
import matplotlib.pyplot as plt

# from plotting import *
from tqdm import tqdm

params = json5.load(open("json/EVT.json5", "r"))

print(params)
