import argparse
import collections
import time

import torch
import utils
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data
import torch.nn.functional as F

import modules
import wandb


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
parser.add_argument('--pixel-scale', type=float, required=True, help='Normalize pixel values in observation.')
parser.add_argument('--shuffle-objects', type=bool, default=False)
parser.add_argument('--use_interactions', type=str, choices=['True', 'False'])
parser.add_argument('--hard_attention', type=str, choices=['True', 'False'], required=True)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--key_query_size', type=int, default=512)
parser.add_argument('--value_size', type=int, default=512)
parser.add_argument('--project', type=str, required=True)
parser.add_argument('--run_id', type=str, default='run-0')

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

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = os.path.join(save_folder, 'log.txt')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.StateTransitionsDataset(
    hdf5_file=args.dataset)
train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

model_args = {
    'embedding_dim': args.embedding_dim,
    'hidden_dim': args.hidden_dim,
    'action_dim': args.action_dim,
    'input_dims': input_shape,
    'num_objects': args.num_objects,
    'sigma': args.sigma,
    'hinge': args.hinge,
    'ignore_action': args.ignore_action,
    'copy_action': args.copy_action,
    'encoder': args.encoder,
    'shuffle_objects': args.shuffle_objects,
    'use_interactions': args.use_interactions == 'True',
    'num_layers': args.num_layers,
}

if args.hard_attention == 'True':
    model_args['key_query_size'] = args.key_query_size
    model_args['value_size'] = args.value_size
    model = modules.ContrastiveSWMHA(**model_args)
else:
    model = modules.ContrastiveSWM(**model_args)

model = model.to(device)

model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)

if args.decoder:
    if args.encoder == 'large':
        decoder = modules.DecoderCNNLarge(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'medium':
        decoder = modules.DecoderCNNMedium(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    elif args.encoder == 'small':
        decoder = modules.DecoderCNNSmall(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
    decoder.apply(utils.weights_init)
    optimizer_dec = torch.optim.Adam(
        decoder.parameters(),
        lr=args.learning_rate)


# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9

wandb.init(
    project=args.project,
    save_code=True,
    name=args.run_id
)

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0
    epoch_metrics = collections.Counter()
    n = 0
    start = time.perf_counter()

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, action, next_obs, _, _, is_terminal = data_batch
        if torch.all(is_terminal).item():
            continue

        obs = obs[~is_terminal]
        action = action[~is_terminal]
        next_obs = next_obs[~is_terminal]

        obs /= args.pixel_scale
        next_obs /= args.pixel_scale
        optimizer.zero_grad()

        if args.decoder:
            optimizer_dec.zero_grad()
            objs = model.obj_extractor(obs)
            state = model.obj_encoder(objs)

            rec = torch.sigmoid(decoder(state))
            loss = F.binary_cross_entropy(
                rec, obs, reduction='sum') / obs.size(0)

            next_state_pred = state + model.transition_model(state, action)
            next_rec = torch.sigmoid(decoder(next_state_pred))
            next_loss = F.binary_cross_entropy(
                next_rec, next_obs,
                reduction='sum') / obs.size(0)
            loss += next_loss
        else:
            loss, metrics = model.contrastive_loss(obs, action, next_obs)
            for key, value in metrics.items():
                epoch_metrics[key] += value * obs.size(0)

        loss.backward()
        train_loss += loss.item() * obs.size(0)
        optimizer.step()
        n += obs.size(0)

        if args.decoder:
            optimizer_dec.step()

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_batch[0])))

        step += 1

    avg_loss = train_loss / n
    record = {key: value / n for key, value in epoch_metrics.items()}
    record['loss'] = avg_loss
    record['epoch'] = epoch
    record['fps'] = n / (time.perf_counter() - start)
    print('====> Epoch: {} Average loss: {:.8f}'.format(
        epoch, avg_loss))

    wandb.log(record)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_file)
