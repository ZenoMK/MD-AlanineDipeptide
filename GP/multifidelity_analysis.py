import argparse
from utils_angles_fixed import *
import GPy
from emukit.model_wrappers import GPyModelWrapper
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils_angles_fixed import *
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays, convert_x_list_to_array
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel


low_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed_low"
high_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed_high"

# prepare test data
test_hists = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/new_data"
mean_histogram, std_histogram, temps, concs, histograms = load_histograms_and_calculate_stats(test_hists)
X_test, Y_test = prepare_2d_gp_data(histograms, temps, concs)

min_vals = Y_test.min(axis=1, keepdims=True)
max_vals = Y_test.max(axis=1, keepdims=True)

# Normalize to 0-100 range along axis 1
normalized_array = (Y_test - min_vals) / (max_vals - min_vals) * 100
Y_test = normalized_array



# prepare training data
mean_hist_low, std_hist_low, temps_low, concs_low, histograms_low = load_histograms_and_calculate_stats(low_dir)
mean_hist_high, std_hist_high, temps_high, concs_high, histograms_high = load_histograms_and_calculate_stats(high_dir)

X_l, Y_l = prepare_2d_gp_data(histograms_low, temps_low, concs_low)
X_h, Y_h = prepare_2d_gp_data(histograms_high, temps_high, concs_high)

X_train, Y_train = convert_xy_lists_to_arrays([X_l, X_h], [Y_l, Y_h])

kernel = GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=1.)+GPy.kern.White(input_dim = 2, variance = 1)
kernels = [GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=1.)+GPy.kern.White(input_dim = 2, variance = 1), 
           GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=1.)+GPy.kern.White(input_dim = 2, variance = 1)]


lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)


lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
lin_mf_model.optimize()

x_test = convert_x_list_to_array([X_test, X_test])
X_test_l = x_test[:len(X_test)]
X_test_h = x_test[len(X_test):]

Y_pred_l, _ = lin_mf_model.predict(X_test_l)
Y_pred_h, _ = lin_mf_model.predict(X_test_h)

Y_test_flat = Y_test.ravel()
Y_pred_l_flat = Y_pred_l.ravel()
Y_pred_h_flat = Y_pred_h.ravel()

rmse_l = mean_squared_error(Y_test_flat, Y_pred_l_flat, squared=False)
mae_l = mean_absolute_error(Y_test, Y_pred_l)

rmse_h = mean_squared_error(Y_test_flat, Y_pred_h_flat, squared=False)
mae_h = mean_absolute_error(Y_test, Y_pred_h)

print(rmse_l, mae_l, rmse_h, mae_h)


high_gp_model = GPy.models.GPRegression(X_h, Y_h, kernel)
high_gp_model.Gaussian_noise.fix(0)

## Fit the GP model
high_gp_model.optimize_restarts(5)

## Compute mean predictions and associated variance

Y_pred_standard, _  = high_gp_model.predict(X_test)
Y_pred_standard_flat = Y_pred_standard.ravel()

rmse_standard = mean_squared_error(Y_test_flat, Y_pred_standard_flat, squared=False)
mae_standard = mean_absolute_error(Y_test, Y_pred_standard)

print(f"standard: {rmse_standard}, {mae_standard}")

print("done")


