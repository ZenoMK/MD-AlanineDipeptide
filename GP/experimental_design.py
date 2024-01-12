# this is a script to run one iteration of

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
import numpy as np
import GPy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils_angles_fixed import *
from main import *

def expdesign(model):
    """
    One iteration of the experimental design loop.
    Inputs:
        model: GP trained on existing data
    Output:
        a new point to sample
    """
    params = ParameterSpace([ContinuousParameter('Temp', 280 , 370),
                    ContinuousParameter('Conc', 0.05, 0.25)])

    model_emukit = GPyModelWrapper(model)
    model_variance = ModelVariance(model=model_emukit)

    expdesign_loop = ExperimentalDesignLoop(model=model_emukit,
                                            space=params,
                                            acquisition=model_variance,
                                            batch_size=1)
    expdesign_loop.run_loop(user_function=dummy, stopping_condition=1)



def dummy(x):
    return x



m,mse = main(kernel_author = "zeno", kernel_number = 1)
expdesign(m)