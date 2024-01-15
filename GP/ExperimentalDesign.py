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
logging.basicConfig(level=logging.DEBUG)
from main import *
from utils_angles_fixed import *

class ExperimentalDesign():
    def __init__(self,dir, max_iters):
        self.dir = dir
        mean_histogram, std_histogram, temps, concs, histograms = load_histograms_and_calculate_stats(self.dir)
        self.temps = temps
        self.concs = concs
        self.histograms = histograms
        self.max_iters = max_iters

        params = ParameterSpace([DiscreteParameter('Temp', [280,290,300,310,320,330,340,350 370]),
                        DiscreteParameter('Conc', [0.05, 0.10,0.15,0.20,0.25])
                                 ])
        design = LatinDesign(params)
        num_data_points = 15
        X = design.get_samples(num_data_points)
        Y = self.obtain_histogram_for_X(X)

        model = GPRegression(X, Y)
        model_emukit = GPyModelWrapper(model)
        model_variance = ModelVariance(model=model_emukit)



        expdesign_loop = ExperimentalDesignLoop(model=model_emukit,
                                                space=params,
                                                acquisition=model_variance,
                                                batch_size=1)
        expdesign_loop.run_loop(user_function=self.obtain_histogram_for_X, stopping_condition=self.max_iters)


    def obtain_histogram_for_X(self, X):
        """
        Function to obtain the correct histogram for a suggested point from our database of simulations
        """
        outputarr = []
        for x in X:
            t = x[0]
            c = x[1]
            for idx, (temp, conc) in enumerate(zip(self.temps, self.concs)):
                if t == temp and c == conc:
                    hist = gaussian_filter(self.histograms[idx], sigma=0.8)
                    outputarr.append(hist)
        return outputarr
