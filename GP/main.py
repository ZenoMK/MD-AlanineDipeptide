import argparse
from utils import *
import GPy


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

    scaler = StandardScaler()
    # scale train data
    X_train_scaled = scaler.fit_transform(X_train)
    # scale test data
    X_test_scaled = scaler.transform(X_test)


    # Define 4D kernel for GP
    """ Define 4D kernel for GP
    Each of the team members can implement and try different kernels under their name by adding numbers
    """
    match kernel_author:
        # example of 1 kernel for Vlad (+ a default case)
        case "vlad":
            match kernel_number:
                case 1: kernel = GPy.kern.Matern32(input_dim=4, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=4, variance=1.)
                case _: kernel = GPy.kern.Matern32(input_dim=4, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=4, variance=1.)
        case _ :
            kernel = GPy.kern.Matern32(input_dim=4, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=4, variance=1.)


    # Create and optimize GP model
    m = GPy.models.GPRegression(X_train_scaled, Y_train, kernel)
    m.optimize(messages=True)

    # Predict on the test set and calculate MSE
    Y_pred, _ = m.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, Y_pred)

    return m, mse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-kernel_author', type=str)
    parser.add_argument('-kernel_number', type=int)
    args = parser.parse_args()

    main(kernel_author=args.kernel_author, kernel_number=args.kernel_number)
