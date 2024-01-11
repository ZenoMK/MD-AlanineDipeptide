# this is a script to run one iteration of

from emukit.test_functions import branin_function
from emukit.core import ParameterSpace, ContinuousParameter

def expdesign(model):
    """
    One iteration of the experimental design loop.
    Inputs:
        model: GP trained on existing data
    Outputs:
        
    """