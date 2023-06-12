import argparse
import io
import pickle
import warnings

import numpy as np
import skimage
import torch
from matplotlib import pyplot as plt

from sklearn import preprocessing, decomposition
from torch import nn
from torch.utils import data

import modules
import utils
import wandb


def emb2str(embedding):
    return [', '.join(f'{value:+1.2f}' for value in obj_emb) for obj_emb in embedding]


def masks_and_reconstruction(masks, reconstructions):
    masks = np.expand_dims(masks, axis=3).repeat(repeats=3, axis=3)
    img = np.concatenate([masks, reconstructions], axis=0)
    img = skimage.transform.rescale(
        skimage.util.montage(img, fill=(255,) * 3, grid_shape=(2, masks.shape[0]), padding_width=3,
                             channel_axis=-1),
        order=0,
        scale=2,
        channel_axis=-1
    )
    return img


def masks_to_image(masks):
    images = np.zeros(shape=(masks.shape[0], masks.shape[2], masks.shape[3], 3), dtype=np.float32)
    colors = utils.get_colors(num_colors=9)
    for image, mask in zip(images, masks):
        for object_id in range(mask.shape[0]):
            image[mask[object_id] == 255] = colors[object_id][:3]

    return (images * 255).astype(np.uint8)


def mpl_to_numpy(fig, dpi):
    with io.BytesIO() as buff:
        fig.savefig(buff, dpi=dpi, format='raw')
        buff.seek(0)
        return np.reshape(np.frombuffer(buff.getvalue(), dtype=np.uint8),
                          newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))


def plot_pca_episode_embeddings(pca_episode_embeddings, title, marker_size=700, history_size=3):
    colors = utils.get_colors(num_colors=9)
    flatten = pca_episode_embeddings.reshape((-1, pca_episode_embeddings.shape[-1]))
    delta = flatten.max(axis=0) - flatten.min(axis=0)
    max_x, max_y = flatten.max(axis=0) + 0.05 * delta
    min_x, min_y = flatten.min(axis=0) - 0.05 * delta

    figures = []
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    for step in range(pca_episode_embeddings.shape[0]):
        prefix = pca_episode_embeddings[:step + 1]
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.tight_layout()
        ax.set_title(title, fontdict={'fontsize': 17}, pad=0)
        ax.set_ylim((min_y, max_y))
        ax.set_xlim((min_x, max_x))
        for prefix_step, objects_embeddings in enumerate(prefix):
            for object_id, object_embedding in enumerate(objects_embeddings):
                if prefix_step == step:
                    alpha = 1
                elif 0 < step - prefix_step <= history_size:
                    alpha = (1 - (step - prefix_step - 1) / history_size) / 3
                else:
                    alpha = 0

                ax.scatter(object_embedding[0], object_embedding[1], s=marker_size, c=[colors[object_id]], alpha=alpha)

        ax.grid()
        figures.append(mpl_to_numpy(fig, dpi=100))
        plt.close()

    return figures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--project', type=str)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--cswm_model_path', type=str, required=True)
    parser.add_argument('--cswm_metadata_path', type=str, required=True)
    parser.add_argument('--decoder_model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_auxiliary', type=str, default=None)
    parser.add_argument('--dataset_episode', type=str, required=True)
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pixel_scale', type=int, default=255)
    parser.add_argument('--title', type=str, required=True)
    parser.add_argument('--history_size', type=int, default=2)

    args = parser.parse_args()

    cswm_meta_file = args.cswm_metadata_path
    cswm_model_file = args.cswm_model_path
    decoder_model_file = args.decoder_model_path

    cswm_args = pickle.load(open(cswm_meta_file, 'rb'))['args']

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

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
        'neg_loss_coef': cswm_args.neg_loss_coef,
    }

    if hasattr(cswm_args, 'edge_actions'):
        cswm_model_args['edge_actions'] = cswm_args.edge_actions == 'True'
    else:
        warnings.warn(f'"edge_actions" parameter is not defined in {cswm_meta_file}')

    if cswm_args.edge_actions == 'True':
        assert cswm_args.attention in (
            'ground_truth',
            'none'), f'Unsupported attention type={cswm_args.attention} for edge_actions={args.edge_actions}.'

    if cswm_args.attention == 'hard':
        cswm_model_args['key_query_size'] = cswm_args.key_query_size
        cswm_model_args['value_size'] = cswm_args.value_size
        cswm_model_args['num_layers'] = cswm_args.num_layers
        cswm = modules.ContrastiveSWMHA(**cswm_model_args).to(device)
    elif cswm_args.attention == 'soft':
        cswm_model_args['key_query_size'] = cswm_args.key_query_size
        cswm_model_args['value_size'] = cswm_args.value_size
        cswm_model_args['num_layers'] = cswm_args.num_layers
        cswm = modules.ContrastiveSWMSA(**cswm_model_args).to(device)
    elif cswm_args.attention == 'ground_truth':
        cswm_model_args['num_layers'] = cswm_args.num_layers
        cswm = modules.ContrastiveSWM(**cswm_model_args).to(device)
    elif cswm_args.attention == 'gnn':
        cswm_model_args['num_layers'] = cswm_args.num_layers
        cswm = modules.ContrastiveSWM(**cswm_model_args).to(device)
    else:
        cswm = modules.ContrastiveSWM(**cswm_model_args).to(device)

    cswm.load_state_dict(torch.load(cswm_model_file))
    cswm = cswm.eval()
    for param in cswm.parameters():
        param.requires_grad = False

    extractor = cswm.obj_extractor
    encoder = nn.Sequential(cswm.obj_extractor, cswm.obj_encoder)
    del cswm

    dataset = utils.StateTransitionsDataset(
        hdf5_file=args.dataset, hdf5_file_auxiliary=args.dataset_auxiliary)
    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    embeddings = []
    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, action, moving_boxes, next_obs, _, _, is_terminal = data_batch
        obs /= args.pixel_scale
        assert cswm_args.num_objects == obs.size()[1], f'num_objects={cswm_args.num_objects}, num_channels={obs.size()[1]}'
        embeddings.append(encoder(obs).detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    episode_dataset = utils.StateTransitionsDataset(hdf5_file=args.dataset_episode)
    ep = 0
    episode_len = len(episode_dataset.experience_buffer[ep]['action'] + 1)
    observations = []
    for step in range(episode_len):
        obs = episode_dataset._get_observation(ep, step)
        observations.append(obs.astype(np.uint8))
    observations.append(episode_dataset._get_observation(ep, episode_len - 1, next_obs=True).astype(np.uint8))
    observations = np.stack(observations, axis=0)
    observations_tensor = torch.tensor(observations, dtype=torch.float32, device=device) / args.pixel_scale
    episode_embeddings = encoder(observations_tensor).detach().cpu().numpy()

    embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
    scaler = preprocessing.StandardScaler().fit(embeddings)
    embeddings_scaled = scaler.transform(embeddings)

    pca = decomposition.PCA(n_components=args.n_components).fit(embeddings_scaled)
    pca_episode_embedding = pca.transform(scaler.transform(episode_embeddings.reshape((-1, episode_embeddings.shape[-1]))))
    pca_episode_embedding = pca_episode_embedding.reshape((*episode_embeddings.shape[:2], args.n_components))

    figures = plot_pca_episode_embeddings(pca_episode_embedding, title=args.title, history_size=args.history_size)
    images = masks_to_image(observations)

    decoder = modules.DecoderMLPChannelWise(
        input_dim=cswm_args.embedding_dim, hidden_dim=cswm_args.hidden_dim, output_size=cswm_args.input_dims[1:]
    )
    decoder.to(device)
    decoder.load_state_dict(torch.load(decoder_model_file))
    decoder = decoder.eval()

    reconstructed_obs = torch.nn.functional.sigmoid(
        decoder(torch.tensor(episode_embeddings, device=device).view(-1, episode_embeddings.shape[-1])))
    reconstructed_obs = reconstructed_obs.view(observations.shape).unsqueeze(-1).detach().cpu().numpy() * 255
    reconstructed_obs = reconstructed_obs.astype(np.uint8).repeat(repeats=3, axis=-1)

    columns = ['images', 'embeddings', 'reconstructions']
    table = wandb.Table(columns=columns)
    for i in range(observations.shape[0]):
        table.add_data(wandb.Image(images[i]), wandb.Image(figures[i]), wandb.Image(masks_and_reconstruction(observations[i], reconstructed_obs[i])))

    wandb.init(
        project=args.project,
        sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        name=args.run_id
    )

    wandb.log({'episode_table': table})
    wandb.finish()
