import argparse
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
parser.add_argument('--decoder-folder', type=str)
parser.add_argument('--pixel-scale', type=float, required=True, help='Normalize pixel values in observation.')
parser.add_argument('--shuffle-objects', type=bool, default=False)
parser.add_argument('--use_interactions', type=str, choices=['True', 'False'])

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
decoder_folder = os.path.join(save_folder, args.decoder_folder)

if not os.path.exists(decoder_folder):
    os.makedirs(decoder_folder)

meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
decoder_file = os.path.join(decoder_folder, 'decoder.pt')
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
    use_interactions=args.use_interaction
).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

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

for epoch in range(1, args.epochs + 1):
    decoder.train()
    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, action, next_obs, reward, state_value = data_batch
        obs /= args.pixel_scale
        next_obs /= args.pixel_scale

        optimizer_dec.zero_grad()
        objs = model.obj_extractor(obs)
        state = model.obj_encoder(objs)

        rec = torch.sigmoid(decoder(state))
        loss = F.binary_cross_entropy(rec, obs)

        loss.backward()
        train_loss += loss.item()
        optimizer_dec.step()

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data_batch[0])
                ))

        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.8f}'.format(
        epoch, avg_loss))

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(decoder.state_dict(), decoder_file)
