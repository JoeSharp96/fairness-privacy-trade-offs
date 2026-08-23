import sys
import argparse
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs='+', required=True)
parser.add_argument("--skews", nargs="+", required=True)
parser.add_argument("--alphas", nargs="+", required=True)
parser.add_argument("--path", required=True)
args = parser.parse_args()

SKEW = args.skews
SEEDS = list(map(int, args.seeds))
ALPHA = args.alphas
PATH = args.path

print(SKEW)
print(SEEDS)
print(ALPHA)
print(PATH)
