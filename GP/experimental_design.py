# this is a script to run one iteration of

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
import numpy as np
import GPy
import os
from utils_angles_fixed import *

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
    expdesign_loop.run_loop(user_function=dummy(), stopping_condition=1)



def dummy(x):
    return x

hist_data_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data_processing/data_processed"
    # prepare training data
mean_hist, std_hist, temps, concs, histograms = load_histograms_and_calculate_stats(hist_data_dir)

X, Y = prepare_4d_gp_data(histograms, temps, concs, COMPRESS_SIZE = 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

scaler = StandardScaler()
    # scale train data
X_train_scaled = scaler.fit_transform(X_train)

m_load = GPy.models.GPRegression(X_train_scaled, Y, initialize=False)
m_load.update_model(False) # do not call the underlying expensive algebra on load
m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
m_load[:] = np.load("../output/models/zeno_1_model_save.npy") # Load the parameters
m_load.update_model(True)

expdesign(m_load)


