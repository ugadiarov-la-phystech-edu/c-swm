import argparse

import skimage.io as io
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
from collections import defaultdict

import modules

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--decoder-folder', type=str)
parser.add_argument('--pixel-scale', type=float, default=1., help='Normalize pixel values in observation.')

args_eval = parser.parse_args()


meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')
decoder_file = os.path.join(args_eval.save_folder, args_eval.decoder_folder, 'decoder.pt')

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 100
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.PathDataset(
    hdf5_file=args.dataset, path_length=args_eval.num_steps)
eval_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Get data sample
obs = eval_loader.__iter__().next()[0]
input_shape = obs[0][0].size()

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
    encoder=args.encoder).to(device)

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

decoder.load_state_dict(torch.load(decoder_file))
decoder.eval()

with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch

        if observations[0].size(0) != args.batch_size:
            continue

        obs = observations[0] / args.pixel_scale

        state = model.obj_encoder(model.obj_extractor(obs))
        reconstruction = torch.sigmoid(decoder(state))
        images = torch.cat([obs, reconstruction], dim=2).permute(0, 3, 2, 1)
        images = images.reshape(-1, *images.size()[-2:]).permute(1, 0, 2).cpu().numpy() * 255
        images = images.astype(np.uint8)
        io.imsave(os.path.join(args_eval.save_folder, args_eval.decoder_folder, 'examples.png'), images)
        break
