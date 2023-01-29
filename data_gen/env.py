"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""
import collections
# Get env directory
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import argparse

# noinspection PyUnresolvedReferences
import envs

import utils

import gym
from gym import logger

import numpy as np
from PIL import Image


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


def crop_normalize(img, crop_ratio):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
    return np.transpose(np.array(img), (2, 0, 1))


def add_observation(use_run_length_encoding, episode_data_dict, observation, next_obs=False):
    obs_key = 'next_obs' if next_obs else 'obs'
    if not use_run_length_encoding:
        episode_data_dict[obs_key].append(observation)
        return

    obs_starts, obs_lengths, obs_values = utils.rlencode(observation.reshape(-1))
    episode_data_dict[f'{obs_key}_starts'].append(obs_starts)
    episode_data_dict[f'{obs_key}_lengths'].append(obs_lengths)
    episode_data_dict[f'{obs_key}_values'].append(obs_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', type=str, default='ShapesTrain-v0',
                        help='Select the environment to run.')
    parser.add_argument('--fname', type=str, default='data/shapes_train.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--atari', action='store_true', default=False,
                        help='Run atari mode (stack multiple frames).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--rle', type=str, choices=['True', 'False'],
                        help='Use run-length encoding for observations')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    use_rle = args.rle is not None and args.rle == 'True'

    env = gym.make(args.env_id)

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    env.seed(args.seed)

    agent = RandomAgent(env.action_space)

    episode_count = args.num_episodes
    reward = 0
    done = False

    crop = None
    warmstart = None
    if args.env_id == 'PongDeterministic-v4':
        crop = (35, 190)
        warmstart = 58
    elif args.env_id == 'SpaceInvadersDeterministic-v4':
        crop = (30, 200)
        warmstart = 50

    if args.atari:
        env._max_episode_steps = warmstart + 11

    replay_buffer = []
    lengths = []

    for i in range(episode_count):
        replay_buffer.append(collections.defaultdict(list))

        ob = env.reset()

        if args.atari:
            # Burn-in steps
            for _ in range(warmstart):
                action = agent.act(ob, reward, done)
                ob, _, _, _ = env.step(action)
            prev_ob = crop_normalize(ob, crop)
            ob, _, _, _ = env.step(0)
            ob = crop_normalize(ob, crop)

            while True:
                add_observation(use_rle, replay_buffer[i], np.concatenate((ob, prev_ob), axis=0))
                prev_ob = ob

                action = agent.act(ob, reward, done)
                ob, reward, done, _ = env.step(action)
                ob = crop_normalize(ob, crop)

                replay_buffer[i]['action'].append(action)
                add_observation(use_rle, replay_buffer[i], np.concatenate((ob, prev_ob), axis=0), next_obs=True)

                if done:
                    break
        else:

            while True:
                add_observation(use_rle, replay_buffer[i], ob[1])

                action = agent.act(ob, reward, done)
                ob, reward, done, _ = env.step(action)

                replay_buffer[i]['action'].append(action)
                add_observation(use_rle, replay_buffer[i], ob[1], next_obs=True)

                if done:
                    lengths.append(len(replay_buffer[i]['action']))
                    break

        if i % 10 == 0:
            mean_length = 0
            if len(lengths) > 0:
                mean_length = sum(lengths) / len(lengths)
            print(f"iter {i}, mean episode length: {mean_length}")

    env.close()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname, use_rle)

