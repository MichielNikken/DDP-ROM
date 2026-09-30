import numpy as np
import einops
from scipy.spatial.transform import Rotation as R

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn

def get_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)

#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#

#------------------------ @TODO: remove some of these ------------------------#

def arctanh_actions(*args, **kwargs):
    epsilon = 1e-4

    def _fn(dataset):
        actions = dataset['actions']
        assert actions.min() >= -1 and actions.max() <= 1, \
            f'applying arctanh to actions in range [{actions.min()}, {actions.max()}]'
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset['actions'] = np.arctanh(actions)
        return dataset

    return _fn

def add_deltas(env):

    def _fn(dataset):
        deltas = dataset['next_observations'] - dataset['observations']
        dataset['deltas'] = deltas
        return dataset

    return _fn
