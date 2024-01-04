from GP.predict_landscape import *
import GPy

def main(train_dir, test_dir, kernel):
    """
    train_dir: dir with training data
    test_dir: dir with test data
    kernel: pre-defined kernel
    """

    # prepare training data
    mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(train_dir)

    X, Y = prepare_4d_gp_data(histograms, temps, concs)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    X_train = X
    Y_train = Y

    # scale train data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # prepare test data
    mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats_test(test_dir)
    X_test, Y_test = prepare_4d_gp_data(histograms, temps, concs)

    # scale test data
    X_test_scaled = scaler.transform(X_test)

    # Define 4D kernel for GP
    ker = GPy.kern.Matern32(input_dim=4, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=4, variance=1.)

    # Create and optimize GP model
    m = GPy.models.GPRegression(X_train_scaled, Y_train, kernel)
    m.optimize(messages=True)

    # Predict on the test set and calculate MSE
    Y_pred, _ = m.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, Y_pred)

    return m, mse
