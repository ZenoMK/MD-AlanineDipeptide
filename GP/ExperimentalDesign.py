# all of this code is an adaption of:
# https://emukit.github.io/experimental-design/#references-on-experimental-design


from emukit.core import ParameterSpace, DiscreteParameter
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
from scipy.ndimage import gaussian_filter
from emukit.test_functions import branin_function
import numpy as np
import GPy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils_angles_fixed import *
from emukit.core.initial_designs.latin_design import LatinDesign
import logging
from main import *
from utils_angles_fixed import *

class ExperimentalDesign():
    logging.basicConfig(level=logging.NOTSET)
    def __init__(self,dir, max_iters):
        self.dir = dir
        mean_histogram, std_histogram, temps, concs, histograms = load_histograms_and_calculate_stats(self.dir)
        self.temps = temps
        self.concs = concs
        self.histograms = histograms
        self.max_iters = max_iters

        params = ParameterSpace([DiscreteParameter('Temp', [280,290,300,310,320,330,340,350, 370]),
                        DiscreteParameter('Conc', [0.05, 0.10,0.15,0.20,0.25])
                                 ])
        design = LatinDesign(params)
        num_data_points = 15
        pts = design.get_samples(num_data_points)
        X,Y = prepare_2d_gp_data(self.histograms, pts[:,0], pts[:,1])

        Y = self.obtain_histogram_for_X(X)
        kernel = GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=1.)
        model = GPy.models.GPRegression(X, Y, kernel=kernel)
        model_emukit = GPyModelWrapper(model)
        model_variance = ModelVariance(model=model_emukit)



        self.expdesign_loop = ExperimentalDesignLoop(model=model_emukit,
                                                space=params,
                                                acquisition=model_variance,
                                                batch_size=1)
    def run_loop(self):
        self.expdesign_loop.run_loop(user_function=self.obtain_histogram_for_X, stopping_condition=self.max_iters)


    def obtain_histogram_for_X(self, X):
        """
        Function to obtain the correct histogram for a suggested point from our database of simulations
        """
        _,Y = prepare_2d_gp_data(self.histograms, X[:,0], X[:,1])
        return Y

    def get_model(self):
        #print(self.expdesign_loop.model)
        return self.expdesign_loop.model