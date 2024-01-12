import argparse
from utils_angles_fixed import *
import GPy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

def main(kernel_author=None, kernel_number=None):
    """
    kernel_author: add your name in match statement below
    kernel_number: add numbered kernels under your name as you experiment w different kernels
    """

    hist_data_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed"
    # prepare training data
    mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(hist_data_dir)

    X, Y = prepare_4d_gp_data(histograms, temps, concs)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


    # only apply scaling to some columns, see: https://stackoverflow.com/questions/38420847/apply-standardscaler-to-parts-of-a-data-set
    ct = ColumnTransformer([
        ('somename', StandardScaler(), [0,1])
    ], remainder='passthrough')

    ct.fit_transform(X_train)
    # scale train data
    X_train_scaled = ct.fit_transform(X_train)
    # scale test data
    X_test_scaled = ct.fit_transform(X_test)


    # Define 4D kernel for GP
    """ Define 4D kernel for GP
    Each of the team members can implement and try different kernels under their name by adding numbers
    """
    match kernel_author:
        # example of 1 kernel for Vlad (+ a default case)
        case "vlad":
            match kernel_number:
                case 1: kernel = GPy.kern.Matern32(input_dim=4, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=4, variance=1.)
                case _: kernel = GPy.kern.White(input_dim=4, variance=1.)
        case "zeno":
            match kernel_number:
                # this will be on GPy without extras
                case 3 : kernel = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)

        case _ :
            kernel = GPy.kern.Matern32(input_dim=4, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=4, variance=1.)

    print(kernel_author)
    # Create and optimize GP model
    m = GPy.models.GPRegression(X_train_scaled, Y_train, kernel)
    m.optimize(messages=True, max_iters=1)
    #m.pickle(f'../output/models/{kernel_author}_{kernel_number}_model_save')
    np.save(f'../output/models/{kernel_author}_{kernel_number}_model_save.npy', m.param_array)

    # Predict on the test set and calculate MSE
    Y_pred, _ = m.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, Y_pred)
    np.save(f'../output/predictions/{kernel_author}_{kernel_number}_model_save.npy', Y_pred)
    print(f"MSE: {mse}")
    return m, mse

main(kernel_author = "zeno", kernel_number = 1)
