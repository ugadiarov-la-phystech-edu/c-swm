import argparse
import math
import pickle
import random

import gym
import numpy as np
import skimage
import torch
from torch import nn

import modules
import wandb

from data_gen.env import RandomAgent
from envs.push import AdHocPushAgent, Push


def emb2str(embedding):
    return [', '.join(f'{value:+1.2f}' for value in obj_emb) for obj_emb in embedding]


def channel_wise_images(observation):
    img = skimage.transform.rescale(
        skimage.util.montage(observation, fill=(255,) * 3, grid_shape=(1, observation.shape[0]), padding_width=5,
                             multichannel=True),
        order=0,
        scale=2
    )
    return img


parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--env_id', type=str, required=True)
parser.add_argument('--project', type=str)
parser.add_argument('--run_id', type=str)
parser.add_argument('--ad_hoc_agent', type=str, choices=['True', 'False'])
parser.add_argument('--random_action_proba', type=float, default=0.5)
parser.add_argument('--cswm_model_path', type=str, required=True)
parser.add_argument('--cswm_metadata_path', type=str, required=True)
parser.add_argument('--reward_model_path', type=str, required=False)
parser.add_argument('--reward_model_metadata_path', type=str, required=False)
parser.add_argument('--state_value_model_path', type=str, required=False)
parser.add_argument('--state_value_model_metadata_path', type=str, required=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.99)

args = parser.parse_args()

cswm_meta_file = args.cswm_metadata_path
cswm_model_file = args.cswm_model_path
reward_model_metadata_file = args.reward_model_metadata_path
reward_model_file = args.reward_model_path
state_value_model_metadata_file = args.state_value_model_metadata_path
state_value_model_file = args.state_value_model_path

cswm_args = pickle.load(open(cswm_meta_file, 'rb'))['args']
reward_args = pickle.load(open(reward_model_metadata_file, 'rb'))['args']
state_value_args = pickle.load(open(state_value_model_metadata_file, 'rb'))['args']

args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

env = gym.make(args.env_id)

np.random.seed(args.seed)
env.action_space.seed(args.seed)
env.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.ad_hoc_agent == 'True':
    assert isinstance(env.unwrapped, Push)
    agent = AdHocPushAgent(env, random_action_proba=args.random_action_proba)
else:
    agent = RandomAgent(env.action_space)

observations = [env.reset()[1]]
actions = []
rewards = [math.nan]
reward = None
done = False

steps = 0
while not done:
    steps += 1
    action = agent.act(observations[-1], reward, done)
    actions.append(action)
    ob, reward, done, _ = env.step(action)
    rewards.append(reward)
    observations.append(ob[1])
    if done:
        env.close()
        break

obs = torch.tensor(observations, dtype=torch.uint8, device=device) / cswm_args.pixel_scale
next_obs = obs[1:]
action = torch.tensor(actions, dtype=torch.int64, device=device)
input_shape = obs[0].size()

cswm_model_args = {
    'embedding_dim': cswm_args.embedding_dim,
    'hidden_dim': cswm_args.hidden_dim,
    'action_dim': cswm_args.action_dim,
    'input_dims': input_shape,
    'num_objects': cswm_args.num_objects,
    'sigma': cswm_args.sigma,
    'hinge': cswm_args.hinge,
    'ignore_action': cswm_args.ignore_action,
    'copy_action': cswm_args.copy_action,
    'encoder': cswm_args.encoder,
    'shuffle_objects': cswm_args.shuffle_objects,
    'use_interactions': cswm_args.use_interactions == 'True',
    'num_layers': cswm_args.num_layers,
}

if cswm_args.hard_attention == 'True':
    cswm_model_args['key_query_size'] = cswm_args.key_query_size
    cswm_model_args['value_size'] = cswm_args.value_size
    cswm = modules.ContrastiveSWMHA(**cswm_model_args).to(device)
    action_converter = modules.ActionConverter(cswm_args.action_dim, attention_module=cswm.attention)
else:
    cswm = modules.ContrastiveSWM(**cswm_model_args).to(device)
    action_converter = modules.ActionConverter(cswm_args.action_dim, attention_module=None)

cswm.load_state_dict(torch.load(cswm_model_file))
cswm = cswm.eval()
for param in cswm.parameters():
    param.requires_grad = False

extractor = cswm.obj_extractor
encoder = nn.Sequential(cswm.obj_extractor, cswm.obj_encoder)
del cswm

reward_use_next_state = reward_args.use_next_state == 'True'
reward_model_input_dim = cswm_args.embedding_dim
if reward_use_next_state:
    reward_model_input_dim = 2 * cswm_args.embedding_dim

reward_model = modules.TransitionGNN(
    input_dim=reward_model_input_dim,
    hidden_dim=reward_args.hidden_dim,
    action_dim=cswm_args.action_dim,
    num_objects=cswm_args.num_objects,
    ignore_action=reward_args.ignore_action,
    copy_action=reward_args.copy_action,
    use_interactions=reward_args.use_interactions == 'True',
    output_dim=1,
).to(device)

reward_model.load_state_dict(torch.load(reward_model_file))
reward_model = reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False

state_value_use_next_state = state_value_args.use_next_state == 'True'
state_value_model_input_dim = cswm_args.embedding_dim
if state_value_use_next_state:
    state_value_model_input_dim = 2 * cswm_args.embedding_dim

state_value_model = modules.TransitionGNN(
    input_dim=state_value_model_input_dim,
    hidden_dim=state_value_args.hidden_dim,
    action_dim=cswm_args.action_dim,
    num_objects=cswm_args.num_objects,
    ignore_action=state_value_args.ignore_action,
    copy_action=state_value_args.copy_action,
    use_interactions=state_value_args.use_interactions == 'True',
    output_dim=1,
).to(device)

state_value_model.load_state_dict(torch.load(state_value_model_file))
state_value_model = state_value_model.eval()
for param in state_value_model.parameters():
    param.requires_grad = False

slots = extractor(obs)
assert obs.size(-1) % slots.size(
    -1) == 0, f'Expected: size of feature map be a factor of observation size. Actual: feature map size = {slots.size(-1)}, observation size = {obs.size(-1)}'
slots = (slots - slots.min()) / (slots.max() - slots.min())
slots = slots.unsqueeze(-1).detach().cpu().numpy() * 255
slots = slots.astype(np.uint8).repeat(repeats=3, axis=-1)

state = encoder(obs)
state_next_state = torch.cat([state[:-1], encoder(next_obs)], dim=-1)

attended_action = action_converter.convert(state[:-1], action)
target_object_ids = action_converter.target_object_id(state[:-1], action).tolist()
step_returns = [np.zeros(shape=(cswm_args.num_objects,), dtype=np.float32)]
for reward, target_object_id in zip(reversed(rewards[1:]), reversed(target_object_ids)):
    step_return = args.gamma * step_returns[-1]
    step_return[target_object_id] += reward
    step_returns.append(step_return)

rewards_object_wise = np.zeros(shape=(len(rewards), cswm_args.num_objects), dtype=np.float32)
rewards_object_wise[1:][np.arange(len(target_object_ids)), target_object_ids] = rewards[1:]
rewards_object_wise[0] = np.nan

step_returns = step_returns[::-1]
step_returns_sum = [step_return.sum() for step_return in step_returns]

state_values = state_value_model([state_next_state if state_value_use_next_state else state, attended_action, False])[
    0].squeeze(-1).detach().cpu().numpy()
state_values_sum = state_values.sum(-1)
state_values = state_values.tolist()
state_values_sum = state_values_sum.tolist()

states_delta = [[]] + (state[1:] - state[:-1]).tolist()
states = state.tolist()

action2symbol = ['🡇', '🡄', '🡅', '🡆']
actions = [f'obj:{a // len(action2symbol)} {action2symbol[a % len(action2symbol)]} | predict:{object_id}' for a, object_id in zip(actions, target_object_ids)] + ['']

obs = obs.detach().cpu().numpy() * 255

if env.channel_wise:
    obs = obs.sum(axis=1, keepdims=True).repeat(repeats=3, axis=1)

obs = obs.transpose([0, 2, 3, 1]).astype(np.uint8)

images = np.concatenate((np.expand_dims(obs, axis=1), slots), axis=1)

columns = ['image', 'action', 'emb', 'emb_delta', 'reward', 'return', 'return_sum', 'value', 'value_sum']
table = wandb.Table(columns=columns)

for image, action, emb, emb_delta, reward, step_return, step_return_sum, value, value_sum in zip(images, actions,
                                                                                                 states, states_delta,
                                                                                                 rewards_object_wise,
                                                                                                 step_returns,
                                                                                                 step_returns_sum,
                                                                                                 state_values,
                                                                                                 state_values_sum):
    emb_str = emb2str(emb)
    emb_delta_str = emb2str(emb_delta)
    table.add_data(wandb.Image(channel_wise_images(image)), action, emb_str, emb_delta_str, reward,
                   step_return, step_return_sum, value, value_sum)

wandb.init(
    project=args.project,
    sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    name=args.run_id
)

wandb.log({'episode_table': table})
wandb.finish()
