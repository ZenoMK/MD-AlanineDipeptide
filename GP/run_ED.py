# this will be used to create and run an expdesign object
from ExperimentalDesign import ExperimentalDesign
import os
import numpy as np

dir = hist_data_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed"
obj = ExperimentalDesign(dir, 45)