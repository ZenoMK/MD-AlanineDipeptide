import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import GPy

COMPRESS_SIZE = 36  # number of bins, i.e. 18 because we want roughly ranges of 10 integer values for angles

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

    for idx, (temp, conc) in enumerate(zip(temps, concs)):
        for i in range(COMPRESS_SIZE):
            for j in range(COMPRESS_SIZE):
                X.append([temp, conc, i, j])
                Y.append(compressed_histograms[idx, i, j])

    return np.array(X), np.array(Y).reshape(-1, 1)


def plot_predicted_landscape_for_temp_conc(m, scaler, temp, conc):
    predicted_landscape = np.zeros((COMPRESS_SIZE, COMPRESS_SIZE))
    for i in range(COMPRESS_SIZE):
        for j in range(COMPRESS_SIZE):
            # Prepare input for prediction
            input_data = np.array([[temp, conc, i, j]])
            input_data_scaled = scaler.transform(input_data)

            # Predict and store the value
            predicted_value, _ = m.predict(input_data_scaled)
            predicted_landscape[i, j] = predicted_value

    # Plotting the predicted landscape as a 2D histogram
    plt.figure(figsize=(8, 6))
    plt.imshow(predicted_landscape, cmap='viridis', origin='lower')
    plt.colorbar(label='Predicted Value')
    plt.xlabel('Bin X')
    plt.ylabel('Bin Y')
    plt.title(f'Predicted Landscape for Temperature {temp} and Concentration {conc}')
    plt.show()


def find_and_plot_compressed_histogram(temps, concs, histograms, temp_val, conc_val, new_size=(COMPRESS_SIZE, COMPRESS_SIZE)):
    # Find the index of the histogram with the specified temperature and concentration
    try:
        idx = next(i for i, (t, c) in enumerate(zip(temps, concs)) if t == temp_val and c == conc_val)
    except StopIteration:
        print("Specified temperature and concentration not found in the dataset.")
        return

    # Compress the found histogram
    compressed_hist = compress_histogram(histograms[idx], new_size=new_size)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(compressed_hist, cmap='viridis', origin='lower')
    plt.colorbar(label='Histogram Value')
    plt.xlabel('Bin X')
    plt.ylabel('Bin Y')
    plt.title(f'Compressed Histogram for Temperature {temp_val} and Concentration {conc_val}')
    plt.show()



"""
TODO: what do we need plot_predicted_landscape_for_temp_conc and find_and_plot_compressed_histogram for? refactor, etc
temps = [310, 320, 330, 330]
concs = [17.5, 12.5, 7.5, 22.5]
#concs = [7.5, 12.5, 17.5, 22.5]

specific_temp = 310  # Example temperature
specific_conc = 17.5   # Example concentration

plot_predicted_landscape_for_temp_conc(m, scaler, specific_temp, specific_conc)
find_and_plot_compressed_histogram(temps, concs, histograms, specific_temp, specific_conc)

"""

