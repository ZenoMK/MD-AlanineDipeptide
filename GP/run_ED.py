# this will be used to create and run an expdesign object
from ExperimentalDesign import ExperimentalDesign
import os
import numpy as np
import logging
from utils_angles_fixed import *
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.DEBUG)

dir = hist_data_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed"

test_hists = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/new_data"
mean_histogram, std_histogram, temps, concs, histograms = load_histograms_and_calculate_stats(test_hists)
X_test, Y_test = prepare_2d_gp_data(histograms, temps, concs)
obj = ExperimentalDesign(dir)
obj.run_loop(5)
m = obj.get_model()
Y_pred, _ = m.predict(X_test)
print(f"MSE: {mean_squared_error(Y_pred,Y_test)} after 5 iterations")
obj.run_loop(1)
m = obj.get_model()
Y_pred, _ = m.predict(X_test)
print(f"RMSE: {mean_squared_error(Y_test, Y_pred, squared=False)} and MAE: {mean_absolute_error(Y_test, Y_pred)}  after 15 iterations")
obj.run_loop(1)
m = obj.get_model()
Y_pred, _ = m.predict(X_test)
print(f"RMSE: {mean_squared_error(Y_test, Y_pred, squared=False)} and MAE: {mean_absolute_error(Y_test, Y_pred)}  after 25 iterations")
X_used = obj.get_sampled_X()
for i in range(len(X_used)):
    print("["+str(i)+","+str(X_used[i][0])+","+str(X_used[i][1])+"],")
