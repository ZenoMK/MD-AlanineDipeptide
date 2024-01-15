import argparse
from utils_angles_fixed import *
import GPy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def prepare_2d_gp_data(histograms, temps, concs):
    compressed_histograms = np.array([compress_histogram(hist) for hist in histograms])
    X = []
    Y = []
    for idx, (temp, conc) in enumerate(zip(temps, concs)):
        X.append([temp, conc])
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
    tick_indices = np.linspace(0, COMPRESS_SIZE - 1, COMPRESS_SIZE, dtype=int)

    tick_labels = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    plt.xticks(tick_indices, tick_labels)
    plt.yticks(tick_indices, tick_labels)

    # axis labels (right order?)
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    plt.colorbar()
    plt.title(f'Predicted Landscape for Temperature {temp} and Concentration {conc}')
    plt.savefig(f'multid_{temp}_{conc}_pred.png', dpi=300)
    plt.show()

    return predicted_value

def plot_compressed_histogram(compressed_hist, temp_val, conc_val):

    # Plotting
    #plt.figure(figsize=(8, 6))
    plt.figure()
    plt.imshow(compressed_hist.T, origin='lower')
    plt.colorbar()
    tick_indices = np.linspace(0, COMPRESS_SIZE - 1, COMPRESS_SIZE, dtype=int)

    tick_labels = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    plt.xticks(tick_indices, tick_labels)
    plt.yticks(tick_indices, tick_labels)

    # axis labels (right order?)
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    plt.title(f'Compressed Histogram for Temperature {temp_val} and Concentration {conc_val}')
    plt.savefig(f'multid_{temp_val}_{conc_val}_hist.png', dpi=300)
    plt.show()
def main(kernel_author=None, kernel_number=None):
    """
    kernel_author: add your name in match statement below
    kernel_number: add numbered kernels under your name as you experiment w different kernels
    """

    hist_data_dir = f"hist_new"
    # prepare training data
    mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(hist_data_dir)

    X, Y = prepare_2d_gp_data(histograms, temps, concs)
    print(X.shape)
    print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
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
    #m.pickle(f'../output/models/{kernel_author}_{kernel_number}_model_save')
    #np.save(f'../output/models/{kernel_author}_{kernel_number}_model_save.npy', m.param_array)

    # Predict on the test set and calculate MSE
    Y_pred, _ = m.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, Y_pred)
    #print(Y_pred.shape)
    #np.save(f'../output/predictions/{kernel_author}_{kernel_number}_model_save.npy', Y_pred)
    print(f"MSE: {mse}")

    Y_test = Y_test.reshape((11, 360, 360))
    for i in range(len(Y_test)):
        # Access each sample
        predicted_histogram = Y_pred[i]
        temp = X_test[i, 0]
        conc = X_test[i, 1]

        predicted_landscape = plot_predicted_landscape_for_temp_conc(m, scaler, temp, conc)
        predicted_landscape = np.where(predicted_landscape < 0, 0, predicted_landscape)
        compressed_landscape = plot_compressed_histogram(Y_test[i], temp,
                                                         conc)
        mse = mean_squared_error(predicted_landscape, compressed_landscape)
        print(f' temp, conc = {temp}, {conc}   mse = {mse}')

    return m, mse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_author', type=str)
    parser.add_argument('--kernel_number', type=int)
    args = parser.parse_args()

    main(kernel_author=args.kernel_author, kernel_number = args.kernel_number)