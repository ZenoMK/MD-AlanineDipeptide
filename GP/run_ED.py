# this will be used to create and run an expdesign object
from ExperimentalDesign import ExperimentalDesign
import os
import numpy as np
import logging
from utils_angles_fixed import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

dir = hist_data_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed"

test_hists = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/new_data"
mean_histogram, std_histogram, temps, concs, histograms = load_histograms_and_calculate_stats(test_hists)
X_test, Y_test = prepare_2d_gp_data(histograms, temps, concs)

RMSE = np.zeros((10,30))
MAE = np.zeros((10,30))
for i in range(10):
    obj = ExperimentalDesign(dir)
    m = obj.get_model()
    Y_pred, _ = m.predict(X_test)
    Y_test_flat = Y_test.ravel()
    Y_pred_flat = Y_pred.ravel()
    rmse = mean_squared_error(Y_test_flat, Y_pred_flat, squared=False)
    mae = mean_absolute_error(Y_test, Y_pred)
    RMSE[i, 0] = rmse
    MAE[i, 0] = mae
    print(RMSE)
    print(MAE)
    for j in range(1,30):
        obj.run_loop(1)
        m = obj.get_model()
        Y_pred, _ = m.predict(X_test)
        Y_test_flat = Y_test.ravel()
        Y_pred_flat = Y_pred.ravel()
        rmse = mean_squared_error(Y_test_flat, Y_pred_flat, squared=False)
        mae = mean_absolute_error(Y_test, Y_pred)
        RMSE[i,j] = rmse
        MAE[i,j] = mae

RMSE_means = np.mean(RMSE, axis = 1)
MAE_means = np.mean(MAE, axis = 1)

print(RMSE_means)
print(MAE_means)

RMSE_std = np.std(RMSE, axis = 1)
MAE_std = np.std(MAE, axis = 1)
idx = [i for i in range(10)]
#fig, ax = plt.figure()
plt.plot(idx, RMSE_means, color = "blue", label = "RMSE")
plt.fill_between(x = idx, y1 = RMSE_means-RMSE_std, y2 = RMSE_means+RMSE_std, color = "blue", alpha = 0.1)

plt.plot(idx, MAE_means, color = "red", label = "MAE")
plt.fill_between(x = idx, y1 = MAE_means-MAE_std,y2 = RMSE_means+RMSE_std, color = "red", alpha = 0.1)

plt.xlabel("Iterations")
plt.legend()
plt.title("Performance metrics during Experimental Design")
plt.savefig("ED_plot")





