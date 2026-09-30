import os
import collections
import numpy as np

# TODO make this parameters

max_episodes = 1
T = 120
#Adjust Nx for latents
Nx = 64
Lx = 22
dt = 0.1
frameskip = 1
max_steps = int(((T / dt) - 1000) / frameskip) - 1
oversampling = 15
episode_reward = 0
count = 0
nr_actuators = 8
nr_sensors = Nx
action_scale = 0.0  # 1.0#0.5
alpha = 0.1
mu = 0.0
sigma = 0.8
offset = 4  # for 8 actuators #8 for 4 actuators
env_name = 'KuramotoSivashinsky'
log = True
parametric = False

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl


# -----------------------------------------------------------------------------#
# -------------------------------- general api --------------------------------#
# -----------------------------------------------------------------------------#

# make it the kuramoto one
# def load_environment(name):
#     env = KuramotoSivashinskyEnv(Nx=Nx, Lx=Lx, dt=dt, T=T, frameskip=frameskip, max_rl_steps=max_steps,
#                                  parametric=parametric, action_scale=action_scale, nr_actuators=nr_actuators,
#                                  nr_sensors=nr_sensors, oversampling=oversampling, alpha=alpha, mu=mu, sigma=sigma,
#                                  offset=offset, eval=False)
#     env.max_episode_steps = env.max_steps
#     env.name = name
#     return env


def get_dataset(env, name):
    #gets the function from ks env
    dataset = env.get_dataset(name)

    return dataset


def sequence_dataset(env, preprocess_fn, name):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env, name)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if env.name is not None and 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


# -----------------------------------------------------------------------------#
# -------------------------------- maze2d fixes -------------------------------#
# -----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
