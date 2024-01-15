# this is a script to run one iteration of

from emukit.core import ParameterSpace, DiscreteParameter
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
from emukit.test_functions import branin_function
import numpy as np
import GPy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils_angles_fixed import *
import logging
logging.basicConfig(level=logging.DEBUG)
from main import *

def expdesign(model):
    """
    One iteration of the experimental design loop.
    Inputs:
        model: GP trained on existing data
    Output:
        a new point to sample
    """
    params = ParameterSpace([DiscreteParameter('Temp', [280,290,300,310,320,330,340,350 370]),
                    DiscreteParameter('Conc', [0.05, 0.10,0.15,0.20,0.25])
                             ])

    model_emukit = GPyModelWrapper(model)
    model_variance = ModelVariance(model=model_emukit)

    f, _ = branin_function()

    expdesign_loop = ExperimentalDesignLoop(model=model_emukit,
                                            space=params,
                                            acquisition=model_variance,
                                            batch_size=1)
    expdesign_loop.run_loop(user_function=f, stopping_condition=1)






m,mse = main("zeno", 1)
expdesign(m)