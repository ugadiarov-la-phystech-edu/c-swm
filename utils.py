"""Utility functions."""

import os
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])
        composite_reward = to_float(self.experience_buffer[ep]['composite_reward'][step])

        return obs, action, next_obs, composite_reward


class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.path_length = path_length

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        rewards = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action'][i]
            reward = float(self.experience_buffer[idx]['reward'][i])
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions, rewards


def observed_colors(num_colors, mode, randomize=True):
    if mode == 'ZeroShot':
        c = np.sort(np.random.uniform(0.0, 1.0, size=num_colors))
    else:
        c = (np.arange(num_colors)) / (num_colors-1)
        if not randomize:
            return c

        diff = 1.0 / (num_colors - 1)
        if mode == 'Train':
            diff = diff / 8.0
        elif mode == 'Test-v1':
            diff = diff / 4.0
        elif mode == 'Test-v2':
            diff = diff / 3.0
        elif mode == 'Test-v3':
            diff = diff / 2.0

        unif = np.random.uniform(-diff+EPS, diff-EPS, size=num_colors)
        unif[0] = abs(unif[0])
        unif[-1] = -abs(unif[-1])

        c = c + unif

    return c


def get_cmap(cmap, mode):
    length = 9
    if cmap == 'Sets':
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Set1')
        else:
            cmap = [plt.get_cmap('Set1'), plt.get_cmap('Set3')]
            length = [9,12]
    else :
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Pastel1')
        else:
            cmap = [plt.get_cmap('Pastel1'), plt.get_cmap('Pastel2')]
            length = [9,8]

    return cmap, length


def unobserved_colors(cmap, num_colors, mode, new_colors=None):
    if mode in ['Train', 'ZeroShotShape']:
        cm, length = get_cmap(cmap, mode)
        weights = np.sort(np.random.choice(length, num_colors, replace=False))
        colors = [cm(i/length) for i in weights]
    else:
        cm, length = get_cmap(cmap, mode)
        cm1, cm2 = cm
        length1, length2 = length
        l = length1 + len(new_colors)
        w = np.sort(np.random.choice(l, num_colors, replace=False))
        colors = []
        weights = []
        for i in w:
            if i < length1:
                colors.append(cm1(i/length1))
                weights.append(i)
            else:
                colors.append(cm2(new_colors[i - length1] / length2))
                weights.append(new_colors[i - length1] + 0.5)

    return colors, weights


def get_colors_and_weights(cmap='Set1', num_colors=9, observed=True,
    mode='Train', new_colors=None, randomize=True):
    """Get color array from matplotlib colormap."""
    if observed:
        c = observed_colors(num_colors, mode, randomize=randomize)
        cm = plt.get_cmap(cmap)

        colors = []
        for i in reversed(range(num_colors)):
            colors.append((cm(c[i])))

        weights = [num_colors - idx
                       for idx in range(num_colors)]
    else:
        colors, weights = unobserved_colors(cmap, num_colors, mode, new_colors)

    return colors, weights