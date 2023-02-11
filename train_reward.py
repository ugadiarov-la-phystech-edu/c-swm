import argparse
import time

import torch
from torch import nn

import utils
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data

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
parser.add_argument('--pretrained_cswm_path', type=str, required=True)

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
meta_file = os.path.join(save_folder, 'reward_model_metadata.pkl')
reward_model_file = os.path.join(save_folder, 'reward_model.pt')
encoder_file = os.path.join(save_folder, 'encoder.pt')
attention_file = os.path.join(save_folder, 'attention.pt')
log_file = os.path.join(save_folder, 'reward_log.txt')

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

reward_model = modules.TransitionGNN(
    input_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    num_objects=args.num_objects,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    use_interactions=args.use_interactions == 'True',
    output_dim=1,
).to(device)
reward_model.apply(utils.weights_init)

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
    cswm = modules.ContrastiveSWMHA(**model_args).to(device)
    action_converter = modules.ActionConverter(args.action_dim, attention_module=cswm.attention)
else:
    cswm = modules.ContrastiveSWM(**model_args).to(device)
    action_converter = modules.ActionConverter(args.action_dim, attention_module=None)


cswm.load_state_dict(torch.load(args.pretrained_cswm_path))
cswm = cswm.eval()
for param in cswm.parameters():
    param.requires_grad = False

encoder = nn.Sequential(cswm.obj_extractor, cswm.obj_encoder)
del cswm

optimizer = torch.optim.Adam(
    reward_model.parameters(),
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
    reward_model.train()
    train_loss = 0
    n = 0
    start = time.perf_counter()

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, action, next_obs, rewards = data_batch
        obs /= args.pixel_scale
        embedding = encoder(obs)
        attended_action = action_converter.convert(embedding, action)
        pred_rewards = reward_model([embedding, attended_action, False])[0].squeeze().sum(-1)
        optimizer.zero_grad()

        loss = nn.functional.mse_loss(pred_rewards, rewards.to(torch.float32).squeeze())
        loss.backward()
        train_loss += loss.item() * obs.size(0)
        optimizer.step()
        n += obs.size(0)

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_batch[0])))

        step += 1

    avg_loss = train_loss / n
    print('====> Epoch: {} Average loss: {:.8f}'.format(
        epoch, avg_loss))

    wandb.log({'epoch': epoch, 'reward_loss': avg_loss, 'fps': n / (time.perf_counter() - start)})

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(reward_model.state_dict(), reward_model_file)
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(action_converter.attention_module.state_dict(), attention_file)
