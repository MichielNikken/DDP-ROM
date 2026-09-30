from collections import namedtuple
import numpy as np
import torch

from .preprocessing import get_preprocess_fn
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

ParameterBatch = namedtuple('ParameterBatch', 'trajectories conditions parameters')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


def load_episodic_dataset(name, preprocess_fn):
    data = np.load(name)
    if 'v' in data:
        obs_key = 'v'
    else:
        obs_key = 'obs'

    if 'mu' in data:
        param_key = 'mu'
    else:
        param_key = 'params'

    for v, mu in zip(data[obs_key], data[param_key]):
        episode = {
            'observations': v,
            'parameters': mu,
        }
        yield episode


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, name, horizon=64,
                 normalizer='LimitsNormalizer', preprocess_fns=None, max_path_length=500,
                 max_n_episodes=25000, termination_penalty=0, use_padding=True, discount=1, param_scale=1, include_params=False):
        preprocess_fns = preprocess_fns or []
        self.name = name
        self.preprocess_fn = get_preprocess_fn(preprocess_fns)
        self.param_scale = param_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_params = include_params
        itr = load_episodic_dataset(name, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1] if hasattr(fields, 'actions') else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize(keys=['observations'])

        print(fields)

    def normalize(self, keys=None):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - horizon, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start + 1):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end] if hasattr(self.fields, 'actions') else np.empty((observations.shape[0],0))

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1, dtype=np.float32)

        if self.include_params:
            # TODO separate changing parameters from constants
            params = self.fields.parameters[path_ind, 0]
            batch = ParameterBatch(trajectories, conditions, params)
        else:
            batch = Batch(trajectories, conditions)

        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
