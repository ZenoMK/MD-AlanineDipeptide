import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import GPy
from sklearn.preprocessing import StandardScaler
from IPython.display import display

# Function to load histograms and calculate mean and std
def load_histograms_and_calculate_stats(directory):
    histograms = []
    temps = []
    concs = []

    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            # Correctly parse the filename to extract temp and conc
            # Assuming filename format: "histogram_tempXXX_concYYY.npy"
            parts = filename.split('_')
            temp = int(parts[1][4:])
            conc = int(parts[2][5:7])
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


def prepare_gp_data(histograms, temps, concs, angle_pair):
    bin_x, bin_y = angle_pair  # Specified angle pair

    # Extracting the values from the specified bin for all temp and conc combinations
    bin_values = histograms[:, bin_x, bin_y]

    # Preparing input data points (X) and target values (Y)
    X = np.array([[temp, conc] for temp, conc in zip(temps, concs)])
    Y = np.array(bin_values).reshape(-1, 1)  # Reshape Y to be a column vector

    return X, Y


directory = 'hist'
mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(directory)

def get_top_std_bins_in_blocks(std_hist, histograms, block_size=45):
    num_blocks = 360 // block_size
    top_bins = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            x_start, x_end = i * block_size, (i + 1) * block_size
            y_start, y_end = j * block_size, (j + 1) * block_size
            block = std_hist[x_start:x_end, y_start:y_end]
            max_pos = np.unravel_index(np.argmax(block), block.shape)
            global_max_pos = (max_pos[0] + x_start, max_pos[1] + y_start)
            top_bins.append(global_max_pos)

    return top_bins

def get_top_mean_bins_in_blocks(mean_hist, histograms, block_size=45):
    num_blocks = 360 // block_size
    top_bins = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            x_start, x_end = i * block_size, (i + 1) * block_size
            y_start, y_end = j * block_size, (j + 1) * block_size
            block = mean_hist[x_start:x_end, y_start:y_end]
            max_pos = np.unravel_index(np.argmax(block), block.shape)
            global_max_pos = (max_pos[0] + x_start, max_pos[1] + y_start)
            top_bins.append(global_max_pos)

    return top_bins


# Define kernel
ker = GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=1.)

# Matern kernel for non-smoothness
matern_kernel = GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=1.)

# White kernel for noise
white_kernel = GPy.kern.White(input_dim=2, variance=1.)

# Combining the kernels
ker = matern_kernel + white_kernel
ker = GPy.kern.RatQuad(input_dim=2, variance=1., lengthscale=1., power=0.5)

# Find top bins
top_bins = get_top_mean_bins_in_blocks(std_hist, histograms)

# Iterate over top bins and perform GP regression
for bin_x, bin_y in top_bins:
    X, Y = prepare_gp_data(histograms, temps, concs, (bin_x, bin_y))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and optimize GP model
    m = GPy.models.GPRegression(X_scaled, Y, ker)
    m.optimize(messages=True)

    # Generate grid for predictions
    temp_min, temp_max = np.min(X_scaled[:, 0]), np.max(X_scaled[:, 0])
    conc_min, conc_max = np.min(X_scaled[:, 1]), np.max(X_scaled[:, 1])
    temp_grid = np.linspace(temp_min, temp_max, 100)
    conc_grid = np.linspace(conc_min, conc_max, 100)
    temp_mesh, conc_mesh = np.meshgrid(temp_grid, conc_grid)
    grid_X = np.vstack([temp_mesh.ravel(), conc_mesh.ravel()]).T

    # Predict on the grid
    mean, _ = m.predict(grid_X)
    mean = mean.reshape(temp_mesh.shape)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(temp_mesh, conc_mesh, mean, cmap='viridis', alpha=0.7)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], Y, c='r', marker='o', label='Training Data')
    ax.set_xlabel('Temperature (scaled)')
    ax.set_ylabel('Concentration (scaled)')
    #ax.set_zlabel('Prediction')
    ax.set_title(f'GP Mean Prediction for Bin ({bin_x}, {bin_y})')
    plt.show()
