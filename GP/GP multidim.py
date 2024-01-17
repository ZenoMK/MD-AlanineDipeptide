import argparse
from utils_angles_fixed import *
import GPy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error
def prepare_2d_gp_data(histograms, temps, concs):
    compressed_histograms = np.array([hist for hist in histograms])
    X = []
    Y = []
    for idx, (temp, conc) in enumerate(zip(temps, concs)):
        X.append([temp, conc])
        compressed_histograms[idx] = gaussian_filter(compressed_histograms[idx], sigma=0.5)
        compressed_histograms[idx] = np.where(compressed_histograms[idx] < 1, 0, compressed_histograms[idx])
        Y.append(compressed_histograms[idx].reshape(-1))

    return np.array(X), np.array(Y)

def plot_predicted_landscape_for_temp_conc(m, scaler, temp, conc):
    predicted_landscape = np.zeros((COMPRESS_SIZE, COMPRESS_SIZE))
    phi = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    psi = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    input_data = np.array([[temp, conc]])
    input_data_scaled = scaler.transform(input_data)
    predicted_value, _ = m.predict(input_data_scaled)
    predicted_value = predicted_value.reshape(COMPRESS_SIZE, COMPRESS_SIZE)
    plt.figure(figsize=(6, 6))
    plt.imshow(predicted_landscape.T,
               origin='lower')
    # plt.tight_layout()

    # axis labels (right order?)
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    plt.colorbar()
    plt.title(f'Predicted Landscape for Temperature {temp} and Concentration {conc}')
    plt.savefig(f'multid_{temp}_{conc}_pred.png', dpi=300)
    plt.show()

    return predicted_value

def plot_histogram(compressed_hist, temp_val, conc_val):
    #compressed_hist = np.where(compressed_hist < 1, 0, compressed_hist)
    plt.figure()
    plt.imshow(compressed_hist.T, origin='lower',norm = LogNorm())
    plt.colorbar()


    # axis labels (right order?)
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    #plt.title(f'Ramachandran plot for Temperature {temp_val} and Concentration {conc_val}')
    plt.savefig(f'multid_{temp_val}_{conc_val}_hist.png', dpi=300)
    #plt.show()


def plot_pred_histogram(compressed_hist, temp_val, conc_val):
    # compressed_hist = np.where(compressed_hist < 1, 0, compressed_hist)
    plt.figure()
    plt.imshow(compressed_hist.T, origin='lower', norm=LogNorm())
    plt.colorbar()

    # axis labels (right order?)
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    # plt.title(f'Ramachandran plot for Temperature {temp_val} and Concentration {conc_val}')
    plt.savefig(f'multid_{temp_val}_{conc_val}_pred.png', dpi=300)
    # plt.show()


def plot_metric(metric_plot, metric_name):
    plt.figure()
    plt.imshow(metric_plot.T, origin='lower')
    plt.colorbar()
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    plt.title(f'Average {metric_name}')
    plt.savefig(f'multifid_metric_name.png', dpi=300)
def main(kernel_author=None, kernel_number=None):
    """
    kernel_author: add your name in match statement below
    kernel_number: add numbered kernels under your name as you experiment w different kernels
    """

    hist_data_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed"
    # prepare training data
    mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(hist_data_dir)

    X_train, Y_train = prepare_2d_gp_data(histograms, temps, concs)

    hist_data_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/new_data"
    # prepare training data
    mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(hist_data_dir)

    X_test, Y_test = prepare_2d_gp_data(histograms, temps, concs)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    scaler = StandardScaler()
    #X_train = X
    #Y_train = Y
    # scale train data
    X_train_scaled = scaler.fit_transform(X_train)
    # scale test data
    X_test_scaled = scaler.transform(X_test)


    # Define 4D kernel for GP
    """ Define 4D kernel for GP
    Each of the team members can implement and try different kernels under their name by adding numbers
    """\


    kernel= GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=2, variance=1.)


    # Create and optimize GP model
    m = GPy.models.GPRegression(X_train_scaled, Y_train, kernel)
    m.optimize(messages=True)
    print(m.kern)
    #m.kern.Mat32.lengthscale = 10
    #m.pickle(f'../output/models/{kernel_author}_{kernel_number}_model_save')
    #np.save(f'../output/models/{kernel_author}_{kernel_number}_model_save.npy', m.param_array)

    # Predict on the test set and calculate MSE
    Y_pred, Y_var = m.predict(X_test_scaled)
    Y_test_flat = Y_test.ravel()
    Y_pred_flat = Y_pred.ravel()

    rmse = mean_squared_error(Y_test_flat, Y_pred_flat, squared=False)
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test_flat, Y_pred_flat)

    #print(Y_pred.shape)
    #np.save(f'../output/predictions/{kernel_author}_{kernel_number}_model_save.npy', Y_pred)
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    plt.figure()
    plt.plot(Y_test_flat)
    plt.figure()
    plt.plot(Y_pred_flat)
    plt.show()

    Y_test = Y_test.reshape((len(Y_test), 360, 360))
    Y_pred = Y_pred.reshape((len(Y_test), 360, 360))
    rmse_plot = np.zeros((360, 360))
    mae_plot = np.zeros((360, 360))
    for i in range(len(Y_test)):
        # Access each sample
        #mse_hist = Y_pred[i] - Y_test[i]
        #plot_histogram(mse_hist, 0,0)
        temp = X_test[i, 0]
        conc = X_test[i, 1]
        rmse_plot += np.square(Y_pred[i] - Y_test[i])
        mae_plot += np.abs(Y_pred[i] - Y_test[i])
        #predicted_landscape = plot_predicted_landscape_for_temp_conc(m, scaler, temp, conc)
        #predicted_landscape = np.where(predicted_landscape < 0, 0, predicted_landscape)
        #plot_histogram(Y_test[i], temp,conc)
        #plot_pred_histogram(Y_pred[i], temp, conc)
        Y_test_flat = Y_test[i].ravel()
        Y_pred_flat = Y_pred[i].ravel()
        rmse = mean_squared_error(Y_test_flat, Y_pred_flat, squared=False)
        mae = mean_absolute_error(Y_pred[i], Y_test[i])
        print(f' temp, conc = {temp}, {conc}   rmse = {rmse}, mae = {mae}')
    rmse_plot = np.sqrt(rmse_plot/len(Y_test))
    mae_plot = mae_plot / len(Y_test)
    plot_histogram(rmse_plot, 0, 0)
    plot_histogram(mae_plot, 0, 0)
    return m, rmse, mae



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_author', type=str)
    parser.add_argument('--kernel_number', type=int)
    args = parser.parse_args()

    main(kernel_author=args.kernel_author, kernel_number = args.kernel_number)
