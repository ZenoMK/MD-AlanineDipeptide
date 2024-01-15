import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import GPy
COMPRESS_SIZE = 360  # number of bins, i.e. 36 because we want roughly ranges of 10 integer values for angles
# Function to load histograms and calculate mean and std
def load_histograms_and_calculate_stats(directory):
    histograms = []
    temps = []
    concs = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            parts = filename.split('_')
            temp = int(parts[1][4:])
            conc = int(parts[2][5:-4])
            data = np.load(os.path.join(directory, filename), allow_pickle=True).item()
            histograms.append(data['histogram'])
            temps.append(temp)
            concs.append(conc)
    histograms = np.array(histograms)
    temps = np.array(temps)
    concs = np.array(concs)
    mean_histogram = np.mean(histograms, axis=0)
    std_histogram = np.std(histograms, axis=0)
    return mean_histogram, std_histogram, temps, concs, histograms
# Function to compress the histogram
def compress_histogram(histogram, new_size=(COMPRESS_SIZE, COMPRESS_SIZE)):
    old_size = histogram.shape
    compression_factor = (old_size[0] // new_size[0], old_size[1] // new_size[1])
    compressed_hist = np.zeros(new_size)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            x_start, x_end = i * compression_factor[0], (i + 1) * compression_factor[0]
            y_start, y_end = j * compression_factor[1], (j + 1) * compression_factor[1]
            compressed_hist[i, j] = np.mean(histogram[x_start:x_end, y_start:y_end])

    return compressed_hist


def prepare_4d_gp_data(histograms, temps, concs):
    compressed_histograms = np.array([compress_histogram(hist) for hist in histograms])
    X = []
    Y = []
    phi = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    psi = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    for idx, (temp, conc) in enumerate(zip(temps, concs)):
        for i in range(COMPRESS_SIZE):
            for j in range(COMPRESS_SIZE):
                X.append([temp, conc, phi[i], psi[j]])
                Y.append(compressed_histograms[idx, i, j])

    return np.array(X), np.array(Y).reshape(-1, 1)

from scipy.ndimage import gaussian_filter
def prepare_2d_gp_data(histograms, temps, concs):
    compressed_histograms = np.array([hist for hist in histograms])
    X = []
    Y = []
    for idx, (temp, conc) in enumerate(zip(temps, concs)):
        X.append([temp, conc])
        compressed_histograms[idx] = gaussian_filter(compressed_histograms[idx], sigma=0.8)
        Y.append(compressed_histograms[idx].reshape(-1))

    return np.array(X), np.array(Y)


def plot_predicted_landscape_for_temp_conc(m, scaler, temp, conc):
    predicted_landscape = np.zeros((COMPRESS_SIZE, COMPRESS_SIZE))
    phi = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    psi = np.arange(-180, 180, 360 / COMPRESS_SIZE)
    for i in range(COMPRESS_SIZE):
        for j in range(COMPRESS_SIZE):
            # Prepare input for prediction
            input_data = np.array([[temp, conc, phi[i], psi[j]]])
            input_data_scaled = scaler.transform(input_data)
            # Predict and store the value
            predicted_value, _ = m.predict(input_data_scaled)
            predicted_landscape[i, j] = predicted_value

    # Plotting the predicted landscape as a 2D histogram
    plt.figure(figsize=(8, 6))
    plt.imshow(predicted_landscape, cmap='viridis', origin='lower')
    plt.imshow(predicted_landscape.T, cmap='viridis', origin='lower')
    plt.colorbar(label='Predicted Value')
    plt.xlabel('Bin X')
    plt.ylabel('Bin Y')
