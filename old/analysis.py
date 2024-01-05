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

# Function to plot histogram (mean or std)
def plot_histogram(hist, title, colorbar=True):
    plt.figure()
    plt.imshow(hist.T, extent=[-180, 180, -180, 180], origin='lower', norm=LogNorm())
    if colorbar:
        plt.colorbar()
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    plt.title(title)
    plt.show()

# Function to plot 2D scatter plot for a specific bin
# Function to find bins with top 10 biggest standard deviations
def find_top_std_bins(std_hist, top_n=10):
    # Flatten the std_hist array and get the indices of the top N values
    flat_indices = np.argsort(std_hist.ravel())[-top_n:]
    # Convert flat indices to 2D indices
    return np.unravel_index(flat_indices, std_hist.shape)

# Modified function to plot 2D scatter plot for specific bins
def plot_2d_bin_function(histograms, temps, concs, std_hist, top_n=30):
    top_bins = find_top_std_bins(std_hist, top_n)
    for bin_x, bin_y in zip(*top_bins):
        bin_values = histograms[:, bin_x, bin_y]

        plt.figure()
        plt.scatter(concs, temps, c=bin_values, cmap='viridis')
        plt.colorbar(label='Bin Value')
        plt.xlabel('Concentration')
        plt.ylabel('Temperature')
        plt.title(f'Bin Value at ({bin_x}, {bin_y}) for Each Temp and Conc')
        plt.show()


def plot_3d_bin_function(histograms, temps, concs, std_hist, top_n=30):
    top_bins = find_top_std_bins(std_hist, top_n)
    for bin_x, bin_y in zip(*top_bins):
        bin_values = histograms[:, bin_x, bin_y]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(concs, temps, bin_values, c=bin_values, cmap='viridis', depthshade=False)
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Temperature')
        ax.set_zlabel('Bin Value')
        ax.set_title(f'3D View - Bin Value at ({bin_x}, {bin_y}) for Each Temp and Conc')
        plt.show()


def plot_max_std_bins_in_blocks(histograms, temps, concs, std_hist, block_size=45):
    # Calculate the number of blocks
    num_blocks = 360 // block_size

    for i in range(num_blocks):
        for j in range(num_blocks):
            # Define the block boundaries
            x_start, x_end = i * block_size, (i + 1) * block_size
            y_start, y_end = j * block_size, (j + 1) * block_size

            # Extract the block from the std_hist
            block = std_hist[x_start:x_end, y_start:y_end]

            # Find the position of the maximum std within this block
            max_pos = np.unravel_index(np.argmax(block), block.shape)

            # Convert local block position to global position
            global_max_pos = (max_pos[0] + x_start, max_pos[1] + y_start)

            # Extract bin values for this position
            bin_values = histograms[:, global_max_pos[0], global_max_pos[1]]

            # 2D scatter plot for the bin
            plt.figure()
            plt.scatter(concs, temps, c=bin_values, cmap='viridis')
            plt.colorbar(label='Bin Value')
            plt.xlabel('Concentration')
            plt.ylabel('Temperature')
            plt.title(f'Bin Value at ({global_max_pos[0]}, {global_max_pos[1]}) for Each Temp and Conc')
            plt.show()



def plot_temp_vs_peak_conc_histogram(histograms, temps, concs):
    peak_concs = []

    # Iterate through each bin position (assuming histograms is a 3D array: [temp, phi, psi])
    for i in range(histograms.shape[1]):  # phi dimension
        for j in range(histograms.shape[2]):  # psi dimension
            for temp_index, temp in enumerate(temps):
                # Extract histogram data for this temperature and bin position
                temp_data = histograms[temp_index, i, j]

                # Find the concentration index corresponding to the peak value for this bin position
                peak_conc_index = np.argmax(temp_data)
                if peak_conc_index < len(concs):
                    peak_concs.append(concs[peak_conc_index])

    # Plotting the histogram
    plt.figure()
    plt.hist(peak_concs, bins=len(concs), color='blue', alpha=0.7)
    plt.xlabel('Peak Concentration')
    plt.ylabel('Frequency')
    plt.title('Histogram of Peak Concentrations Across Temperatures')
    plt.show()


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

plot_histogram(mean_hist, 'Mean Histogram')
plot_histogram(std_hist, 'Standard Deviation Histogram')
plot_max_std_bins_in_blocks(histograms, temps, concs, std_hist)

#84 224 84 248
X, Y = prepare_gp_data(histograms, temps, concs, (84, 224))

scaler = StandardScaler()
X = scaler.fit_transform(X)
# define kernel
ker = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)

# create simple GP model
m = GPy.models.GPRegression(X,Y,ker)

# optimize and plot
m.optimize(messages=True)
mean, std = m.predict(X)
# Scatter plot the training data in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y, c='r', marker='o', label='Training Data')

# Generate a grid for predictions
temp_min, temp_max = np.min(X[:, 0]), np.max(X[:, 0])
conc_min, conc_max = np.min(X[:, 1]), np.max(X[:, 1])
temp_grid = np.linspace(temp_min, temp_max, 100)
conc_grid = np.linspace(conc_min, conc_max, 100)
temp_mesh, conc_mesh = np.meshgrid(temp_grid, conc_grid)
grid_X = np.vstack([temp_mesh.ravel(), conc_mesh.ravel()]).T

# Make predictions on the grid
mean, _ = m.predict(grid_X)
mean = mean.reshape(temp_mesh.shape)

# Plot the mean predictions as a 3D surface
ax.plot_surface(temp_mesh, conc_mesh, mean, cmap='viridis', alpha=0.7)

ax.set_xlabel('Temperature')
ax.set_ylabel('Concentration')
ax.set_zlabel('GP Prediction')
ax.set_title('Gaussian Process Mean Prediction (3D)')

plt.legend()
plt.show()
