import argparse
import collections

import torch
import utils
import datetime
import os

import numpy as np
import logging

from torch.utils import data

import modules
import cv2 as cv


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='small',
                    help='Object extrator CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')

parser.add_argument('--decoder', action='store_true', default=False,
                    help='Train model using decoder and pixel-based loss.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--pixel-scale', type=float, default=1., help='Normalize pixel values in observation.')
parser.add_argument('--shuffle-objects', type=bool, default=False)
parser.add_argument('--interaction_score_threshold', type=float, required=True)
parser.add_argument('--l1_loss_coef', type=float, required=True)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)
high_score_obs_folder = save_folder + '/high_score'
low_score_obs_folder = save_folder + '/low_score'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
os.makedirs(high_score_obs_folder, exist_ok=True)
os.makedirs(low_score_obs_folder, exist_ok=True)

meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = os.path.join(save_folder, 'log.txt')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.StateTransitionsDataset(
    hdf5_file=args.dataset)
train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    encoder=args.encoder, shuffle_objects=args.shuffle_objects,
    interaction_score_threshold=args.interaction_score_threshold,
    l1_loss_coef=args.l1_loss_coef,
).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

high_score_obs = collections.defaultdict(list)
low_score_obs = collections.defaultdict(list)

for epoch in range(1):
    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, action, next_obs = data_batch
        obs /= args.pixel_scale
        next_obs /= args.pixel_scale

        objs = model.obj_extractor(obs)
        state = model.obj_encoder(objs)
        pred_transition, info = model.transition_model(state, action)
        high_score_ids = info['high_score_ids'].detach().cpu().numpy()
        high_scores = info['high_scores'].detach().cpu().numpy()

        log_info = {}
        to_log = False
        for ids, score in zip(high_score_ids, high_scores):
            obs_id = ids[0]
            if 'obs_id' not in log_info or obs_id != log_info['obs_id']:
                if to_log:
                    print(f'{log_info}')
                    to_log = False
                log_info = {'obs_id': obs_id}

            object_ids = (ids[1], ids[2])
            if object_ids[0] > object_ids[1]:
                continue

            log_info[object_ids] = score
            if score > 0.9 and len(high_score_obs[object_ids]) < 10:
                high_score_obs[object_ids].append({'score': score, 'obs': (obs[obs_id].cpu().numpy() * args.pixel_scale).astype(np.uint8), 'batch_idx': batch_idx, 'obs_id': obs_id})
                to_log = True
            elif score < 0.1 and len(low_score_obs[object_ids]) < 10:
                low_score_obs[object_ids].append({'score': score, 'obs': (obs[obs_id].cpu().numpy() * args.pixel_scale).astype(np.uint8), 'batch_idx': batch_idx, 'obs_id': obs_id})
                to_log = True

for object_ids, obs_infos in high_score_obs.items():
    for obs_info in obs_infos:
        batch_idx = obs_info['batch_idx']
        obs_id = obs_info['obs_id']
        score = obs_info['score']
        obs = np.transpose(obs_info['obs'], axes=(1, 2, 0))
        path = f'{high_score_obs_folder}/batchIdx-{batch_idx}_obsId-{obs_id}_objects-{object_ids}_score-{score:.2f}.png'
        cv.imwrite(path, obs)

for object_ids, obs_infos in low_score_obs.items():
    for obs_info in obs_infos:
        batch_idx = obs_info['batch_idx']
        obs_id = obs_info['obs_id']
        score = obs_info['score']
        obs = np.transpose(obs_info['obs'], axes=(1, 2, 0))
        path = f'{low_score_obs_folder}/batchIdx-{batch_idx}_obsId-{obs_id}_objects-{object_ids}_score-{score:.2f}.png'
        cv.imwrite(path, obs)

