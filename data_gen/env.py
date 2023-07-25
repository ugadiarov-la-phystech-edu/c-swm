"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""
import collections
# Get env directory
import sys
from pathlib import Path

from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from envs.cw_envs import CwTargetEnv
from envs.push import Push, AdHocPushAgent

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
    parser.add_argument('--env_type', type=str, choices=['gym', 'cw'], default='gym',
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
    parser.add_argument('--ad_hoc_agent', type=str, choices=['True', 'False'])
    parser.add_argument('--random_action_proba', type=float, default=0.5)
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    use_rle = args.rle is not None and args.rle == 'True'

    if args.env_type == 'gym':
        env = gym.make(args.env_id)
        env.seed(args.seed)
    elif args.env_type == 'cw':
        config = OmegaConf.load(f'envs/config/{args.env_id}.yaml')
        env = CwTargetEnv(config, args.seed)
        env = TimeLimit(env, env.unwrapped._max_episode_length)
    else:
        raise ValueError(f'Unknown environment: type={args.env_type} env_id={args.env_id}')

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)

    if args.ad_hoc_agent == 'True':
        assert isinstance(env.unwrapped, Push)
        agent = AdHocPushAgent(env, args.random_action_proba)
    else:
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
    successes = []
    image_shape = None

    for i in range(episode_count):
        replay_buffer.append(collections.defaultdict(list))

        ob = env.reset()

        if args.atari:
            # Burn-in steps
            image_shape = ob.shape
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
                replay_buffer[i]['reward'].append(reward)
                add_observation(use_rle, replay_buffer[i], np.concatenate((ob, prev_ob), axis=0), next_obs=True)

                if done:
                    break
        else:
            if args.env_type == 'cw':
                ob = ob
            else:
                ob = ob[1]

            image_shape = ob.shape
            is_success = False
            while True:
                add_observation(use_rle, replay_buffer[i], ob)

                action = agent.act(ob, reward, done)
                ob, reward, done, info = env.step(action)
                if args.env_type == 'cw':
                    ob = ob
                else:
                    ob = ob[1]

                replay_buffer[i]['action'].append(action)
                replay_buffer[i]['reward'].append(reward)
                if Push.MOVING_BOXES_KEY in info:
                    replay_buffer[i][Push.MOVING_BOXES_KEY].append(info[Push.MOVING_BOXES_KEY])
                add_observation(use_rle, replay_buffer[i], ob, next_obs=True)
                if not is_success and 'is_success' in info:
                    is_success = info['is_success']

                if done:
                    lengths.append(len(replay_buffer[i]['action']))
                    successes.append(int(is_success))
                    break

        if i % 10 == 0 or i == episode_count - 1:
            mean_length = 0
            success_rate = 0
            if len(lengths) > 0:
                mean_length = sum(lengths) / len(lengths)
                success_rate = sum(successes) / len(successes)
            print(f"iter {i}, mean episode length: {mean_length}, success rate: {success_rate}", flush=True)

    env.close()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname, use_rle, image_shape)

