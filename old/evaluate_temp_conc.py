import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import GPy

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

# Function to prepare GP data
def prepare_gp_data(histograms, temps, concs, angle_pair):
    bin_x, bin_y = angle_pair
    bin_values = histograms[:, bin_x, bin_y]
    X = np.array([[temp, conc] for temp, conc in zip(temps, concs)])
    Y = np.array(bin_values).reshape(-1, 1)
    return X, Y

# Function to get top mean bins in blocks
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

# Load histograms and calculate stats
directory = 'hist'
mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(directory)
top_bins = get_top_mean_bins_in_blocks(mean_hist, histograms)

# Define kernel
ker = GPy.kern.RatQuad(input_dim=2, variance=1., lengthscale=1., power=0.5)

# Open a file to write log details
with open('log.txt', 'w') as log_file:

    # Iterate over top bins and perform GP regression
    for bin_x, bin_y in top_bins:
        X, Y = prepare_gp_data(histograms, temps, concs, (bin_x, bin_y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and optimize GP model
        m = GPy.models.GPRegression(X_train_scaled, Y_train, ker)
        m.optimize(messages=True)

        # Write optimized kernel parameters to log file
        log_file.write(f"Optimized kernel parameters for Bin ({bin_x}, {bin_y}):\n")
        log_file.write(str(m.kern) + '\n')

        # Predict on the test set and calculate MSE
        Y_pred, _ = m.predict(X_test_scaled)
        mse = mean_squared_error(Y_test, Y_pred)
        log_file.write(f"Mean Squared Error for Bin ({bin_x}, {bin_y}): {mse}\n\n")

        # Generate grid for predictions
        temp_min, temp_max = np.min(X_train_scaled[:, 0]), np.max(X_train_scaled[:, 0])
        conc_min, conc_max = np.min(X_train_scaled[:, 1]), np.max(X_train_scaled[:, 1])
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
        ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], Y_train, c='b', marker='o', label='Training Data')
        ax.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], Y_test, c='r', marker='^', label='Test Data')
        ax.set_xlabel('Temperature (scaled)')
        ax.set_ylabel('Concentration (scaled)')
        ax.set_title(f'GP Mean Prediction for Bin ({bin_x}, {bin_y})')
        plt.legend()
        plt.show()
