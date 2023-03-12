import argparse
import time
import warnings

import torch
from torch import nn
from torch.nn import functional

import utils
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data

import modules
import wandb


def compute_returns(dataset, encoder, action_converter, cswm_args, device):
    for episode_id in range(len(dataset.experience_buffer)):
        action = torch.tensor(dataset.experience_buffer[episode_id]['action'], device=device)
        obs = []
        for i in range(action.size()[0]):
            obs.append(dataset._get_observation(episode_id, i) / cswm_args.pixel_scale)

        obs = torch.tensor(obs, device=device)
        embedding = encoder(obs)
        target_object_id = action_converter.target_object_id(embedding, action).detach().cpu().numpy()
        reward = dataset.experience_buffer[episode_id]['reward']
        reward_object_wise = np.zeros(shape=(len(reward), cswm_args.num_objects))
        reward_object_wise[np.arange(0, reward_object_wise.shape[0]), target_object_id] = reward
        dataset.experience_buffer[episode_id]['reward'] = reward_object_wise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='Learning rate.')

    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Number of hidden units in transition MLP.')
    parser.add_argument('--action-dim', type=int, default=4,
                        help='Dimensionality of action space.')
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
    parser.add_argument('--use_interactions', type=str, choices=['True', 'False'])
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--run_id', type=str, default='run-0')
    parser.add_argument('--pretrained_cswm_path', type=str, required=True)
    parser.add_argument('--use_next_state', type=str, choices=['True', 'False'], required=True)
    parser.add_argument('--signal', type=str, choices=['reward', 'return'], required=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)

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
    meta_file = os.path.join(save_folder, f'{args.model_name}_model_metadata.pkl')
    model_file = os.path.join(save_folder, f'{args.model_name}_model.pt')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    attention_file = os.path.join(save_folder, 'attention.pt')
    log_file = os.path.join(save_folder, f'{args.model_name}_log.txt')

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file, 'a'))
    print = logger.info

    pickle.dump({'args': args}, open(meta_file, "wb"))

    device = torch.device('cuda' if args.cuda else 'cpu')

    cswm_meta_file = os.path.join(args.pretrained_cswm_path, 'metadata.pkl')
    cswm_model_file = os.path.join(args.pretrained_cswm_path, 'model.pt')
    cswm_args = pickle.load(open(cswm_meta_file, 'rb'))['args']

    if args.action_dim != cswm_args.action_dim:
        warnings.warn(f'args.action_dim={args.action_dim} cswm_args.action_dim={cswm_args.action_dim}')

    use_next_state = args.use_next_state == 'True'
    reward_model_input_dim = cswm_args.embedding_dim
    if use_next_state:
        reward_model_input_dim = 2 * cswm_args.embedding_dim

    model = modules.TransitionGNN(
        input_dim=reward_model_input_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        num_objects=cswm_args.num_objects,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        use_interactions=args.use_interactions == 'True',
        output_dim=1,
    ).to(device)
    model.apply(utils.weights_init)

    cswm_model_args = {
        'embedding_dim': cswm_args.embedding_dim,
        'hidden_dim': cswm_args.hidden_dim,
        'action_dim': cswm_args.action_dim,
        'input_dims': cswm_args.input_dims,
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

    encoder = nn.Sequential(cswm.obj_extractor, cswm.obj_encoder)
    del cswm

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate)

    dataset = utils.StateTransitionsDataset(
        hdf5_file=args.dataset, gamma=args.gamma)

    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

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
        n = 0
        start = time.perf_counter()

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            obs, action, next_obs, rewards, returns, is_terminal = data_batch
            obs /= cswm_args.pixel_scale
            next_obs /= cswm_args.pixel_scale
            if args.signal == 'reward' or use_next_state:
                if torch.all(is_terminal).item():
                    continue

                obs = obs[~is_terminal]
                action = action[~is_terminal]
                next_obs = next_obs[~is_terminal]
                rewards = rewards[~is_terminal]
                returns = returns[~is_terminal]

            embedding = encoder(obs)
            if args.signal == 'reward':
                attended_action = action_converter.convert(embedding, action)[:, :, :model.action_dim]
                target_object_id = action_converter.target_object_id(embedding, action).detach()
                rewards = rewards.squeeze().to(torch.float32)
                reward_object_wise = torch.zeros(rewards.size()[0], cswm_args.num_objects, dtype=torch.float32, device=device)
                reward_object_wise[torch.arange(0, rewards.size()[0], device=device), target_object_id] = rewards
                ground_truth = reward_object_wise.detach()
            else:
                ground_truth = returns.to(torch.float32).squeeze()
                assert args.ignore_action
                attended_action = torch.zeros_like(action)

            if use_next_state:
                next_embedding = encoder(next_obs)
                embedding = torch.cat([embedding, next_embedding], dim=-1)

            if args.signal == 'reward' or torch.count_nonzero(is_terminal).item() == 0:
                prediction = model([embedding, attended_action, False])[0].squeeze(dim=-1)
                loss = functional.mse_loss(prediction, ground_truth)
            else:
                is_not_terminal = ~is_terminal
                prediction_not_terminal = model([embedding[is_not_terminal], attended_action[is_not_terminal], False])[
                    0].squeeze(dim=-1)
                loss_not_terminal = functional.mse_loss(prediction_not_terminal, ground_truth[is_not_terminal],
                                                        reduction='none')
                prediction_terminal = model([embedding[is_terminal], attended_action[is_terminal], False])[0].squeeze(
                    dim=-1)
                ground_truth_terminal = ground_truth[is_terminal].unsqueeze(-1).expand_as(prediction_terminal)

                loss_terminal = functional.mse_loss(prediction_terminal, ground_truth_terminal, reduction='none')

                loss = loss_not_terminal + loss_terminal

            optimizer.zero_grad()
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
            torch.save(model.state_dict(), model_file)
            torch.save(encoder.state_dict(), encoder_file)
            if action_converter.attention_module is not None:
                torch.save(action_converter.attention_module.state_dict(), attention_file)


if __name__ == '__main__':
    main()
