"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""
import collections
# Get env directory
import sys
from pathlib import Path

from envs.push import Push
from envs.states_generator import StatesGenerator

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import argparse

# noinspection PyUnresolvedReferences
import envs

import utils

import gym
from gym import logger

import numpy as np


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


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
    parser.add_argument('--num_states', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--rle', type=str, choices=['True', 'False'],
                        help='Use run-length encoding for observations')
    parser.add_argument('--extended', type=str, choices=['True', 'False'], default='False')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    use_rle = args.rle is not None and args.rle == 'True'

    env = gym.make(args.env_id)

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    env.seed(args.seed)
    image_shape = None
    states_generator = StatesGenerator(args.seed, env.unwrapped)

    num_states = args.num_states
    replay_buffer = []
    lengths = []

    agent = RandomAgent(env.action_space)
    env.reset()

    episode = 0
    replay_buffer.append(collections.defaultdict(list))
    for _ in range(num_states):
        states_generator.agent_box_goal()
        ob = env.unwrapped._get_observation()
        image_shape = ob[1].shape
        add_observation(use_rle, replay_buffer[episode], ob[1])

        action = agent.act(ob, None, None)
        ob, reward, done, info = env.step(action)

        replay_buffer[episode]['action'].append(action)
        replay_buffer[episode]['reward'].append(reward)
        replay_buffer[episode][Push.MOVING_BOXES_KEY].append(info[Push.MOVING_BOXES_KEY])
        add_observation(use_rle, replay_buffer[episode], ob[1], next_obs=True)

    episode = 1
    replay_buffer.append(collections.defaultdict(list))
    for _ in range(num_states):
        states_generator.agent_box_box()
        ob = env.unwrapped._get_observation()
        image_shape = ob[1].shape
        add_observation(use_rle, replay_buffer[episode], ob[1])

        action = agent.act(ob, None, None)
        ob, reward, done, info = env.step(action)

        replay_buffer[episode]['action'].append(action)
        replay_buffer[episode]['reward'].append(reward)
        replay_buffer[episode][Push.MOVING_BOXES_KEY].append(info[Push.MOVING_BOXES_KEY])
        add_observation(use_rle, replay_buffer[episode], ob[1], next_obs=True)

    if args.extended == 'True':
        episode = 2
        replay_buffer.append(collections.defaultdict(list))
        for i in range(num_states):
            vector = states_generator.agent_goal()
            ob = env.unwrapped._get_observation()
            image_shape = ob[1].shape
            add_observation(use_rle, replay_buffer[episode], ob[1])

            action = env.direction2action[vector]
            if i % 2 == 0:
                while True:
                    agent_action = agent.act(ob, None, None)
                    if action != agent_action:
                        action = agent_action
                        break
            ob, reward, done, info = env.step(action)

            replay_buffer[episode]['action'].append(action)
            replay_buffer[episode]['reward'].append(reward)
            replay_buffer[episode][Push.MOVING_BOXES_KEY].append(info[Push.MOVING_BOXES_KEY])
            add_observation(use_rle, replay_buffer[episode], ob[1], next_obs=True)

        episode = 3
        replay_buffer.append(collections.defaultdict(list))
        for i in range(num_states):
            vector = states_generator.agent_border()
            ob = env.unwrapped._get_observation()
            image_shape = ob[1].shape
            add_observation(use_rle, replay_buffer[episode], ob[1])

            action = env.direction2action[vector]
            if i % 2 == 0:
                while True:
                    agent_action = agent.act(ob, None, None)
                    if action != agent_action:
                        action = agent_action
                        break
            ob, reward, done, info = env.step(action)

            replay_buffer[episode]['action'].append(action)
            replay_buffer[episode]['reward'].append(reward)
            replay_buffer[episode][Push.MOVING_BOXES_KEY].append(info[Push.MOVING_BOXES_KEY])
            add_observation(use_rle, replay_buffer[episode], ob[1], next_obs=True)

        episode = 4
        replay_buffer.append(collections.defaultdict(list))
        for i in range(num_states):
            vector = states_generator.agent_box_border()
            ob = env.unwrapped._get_observation()
            image_shape = ob[1].shape
            add_observation(use_rle, replay_buffer[episode], ob[1])

            action = env.direction2action[vector]
            if i % 2 == 0:
                while True:
                    agent_action = agent.act(ob, None, None)
                    if action != agent_action:
                        action = agent_action
                        break
            ob, reward, done, info = env.step(action)

            replay_buffer[episode]['action'].append(action)
            replay_buffer[episode]['reward'].append(reward)
            replay_buffer[episode][Push.MOVING_BOXES_KEY].append(info[Push.MOVING_BOXES_KEY])
            add_observation(use_rle, replay_buffer[episode], ob[1], next_obs=True)

        episode = 5
        replay_buffer.append(collections.defaultdict(list))
        for i in range(num_states):
            vector = states_generator.agent_box_box_stack()
            ob = env.unwrapped._get_observation()
            image_shape = ob[1].shape
            add_observation(use_rle, replay_buffer[episode], ob[1])

            action = env.direction2action[vector]
            if i % 2 == 0:
                while True:
                    agent_action = agent.act(ob, None, None)
                    if action != agent_action:
                        action = agent_action
                        break
            ob, reward, done, info = env.step(action)

            replay_buffer[episode]['action'].append(action)
            replay_buffer[episode]['reward'].append(reward)
            replay_buffer[episode][Push.MOVING_BOXES_KEY].append(info[Push.MOVING_BOXES_KEY])
            add_observation(use_rle, replay_buffer[episode], ob[1], next_obs=True)

    env.close()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname, use_rle, image_shape)
