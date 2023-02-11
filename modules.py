import utils

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, hinge=1., sigma=0.5, encoder='large',
                 ignore_action=False, copy_action=False, shuffle_objects=False, use_interactions=True, num_layers=3):
        super(ContrastiveSWM, self).__init__()
        self.shuffle_objects = shuffle_objects

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.use_interactions = use_interactions
        self.num_layers = num_layers
        
        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
                shuffle_objects=self.shuffle_objects
            )
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
                shuffle_objects=self.shuffle_objects
            )
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
                shuffle_objects=self.shuffle_objects
            )
        elif encoder == 'identity':
            self.obj_extractor = nn.Identity()
        else:
            raise ValueError(f'Unexpected encoder type: {encoder}')

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects)

        self.transition_model = TransitionGNN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action,
            use_interactions=self.use_interactions,
            num_layers=self.num_layers,
        )

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
            return norm * diff.pow(2).mean(2)
        else:
            pred_state = self.forward_transition(state, action)
            diff = pred_state - next_state
            return norm * diff.pow(2).mean(dim=(1, 2))

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):
        state, next_state = self.extract_objects_(obs, next_obs)

        self.pos_loss = self.energy(state, action, next_state).mean()

        # Sample negative state across episodes at random
        neg_obs, neg_state = self.create_negatives_(obs, state)

        self.neg_loss = self.negative_loss_(state, neg_state)

        loss = self.pos_loss + self.neg_loss

        return loss, {'transition_loss': self.pos_loss.item(), 'contrastive_loss': self.neg_loss.item()}

    def extract_objects_(self, obs, next_obs):
        state = self.forward(obs)
        next_state = self.forward(next_obs)

        return state, next_state

    def create_negatives_(self, obs, state):
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_obs = obs[perm]
        neg_state = state[perm]

        return neg_obs, neg_state

    def negative_loss_(self, state, neg_state):
        energy = self.energy(state, None, neg_state, no_trans=True)
        zeros = torch.zeros_like(energy)
        return torch.max(zeros, self.hinge - energy).mean()

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))

    def forward_transition(self, state, action):
        pred_trans, _, _ = self.transition_model([state, action, False])
        return state + pred_trans


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects, ignore_action=False, copy_action=False,
                 act_fn='relu', layer_norm=True, num_layers=3, use_interactions=True, output_dim=None):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if self.output_dim is None:
            self.output_dim = self.input_dim

        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.use_interactions = use_interactions
        self.num_layers = num_layers

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        tmp_action_dim = self.action_dim
        edge_mlp_input_size = self.input_dim * 2

        self.edge_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            edge_mlp_input_size, self.hidden_dim, act_fn, layer_norm
        ))

        if self.num_objects == 1 or not self.use_interactions:
            node_input_dim = self.input_dim + tmp_action_dim
        else:
            node_input_dim = hidden_dim + self.input_dim + tmp_action_dim

        self.node_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            node_input_dim, self.output_dim, act_fn, layer_norm
        ))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, action=None):
        if action is None:
            x = [source, target]
        else:
            x = [source, target, action]

        out = torch.cat(x, dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)
            self.edge_list = self.edge_list.to(device)

        return self.edge_list

    def process_action_(self, action, viz=False):
        if self.copy_action:
            if len(action.shape) == 1:
                # action is an integer
                action_vec = utils.to_one_hot(action, self.action_dim).repeat(1, self.num_objects)
            else:
                # action is a vector
                action_vec = action.repeat(1, self.num_objects)

            # mix node and batch dimension
            action_vec = action_vec.reshape(-1, self.action_dim).float()
        else:
            # we have a separate action for each node
            if len(action.shape) == 1:
                # index for both object and action
                action_vec = utils.to_one_hot(action, self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)
            else:
                action_vec = action.reshape(action.size(0), self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)

        return action_vec

    def forward(self, x):
        states, action, viz = x

        device = states.device
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.reshape(-1, self.input_dim)

        if not self.ignore_action:
            action_vec = self.process_action_(action, viz=viz)

        edge_attr = None
        edge_index = None

        if num_nodes > 1 and self.use_interactions:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, device)

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col])

        if not self.ignore_action:
            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        # we return the same thing as input but with a changed state
        # this allows us to stack GNNs
        return node_attr, action, viz

    def make_node_mlp_layers_(self, input_dim, output_dim, act_fn, layer_norm):
        layers = []

        for idx in range(self.num_layers):

            if idx == 0:
                # first layer, input_dim => hidden_dim
                layers.append(nn.Linear(input_dim, self.hidden_dim))
                layers.append(utils.get_act_fn(act_fn))
            elif idx == self.num_layers - 2:
                # layer before the last, add layer norm
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(self.hidden_dim))
                layers.append(utils.get_act_fn(act_fn))
            elif idx == self.num_layers - 1:
                # last layer, hidden_dim => output_dim and no activation
                layers.append(nn.Linear(self.hidden_dim, output_dim))
            else:
                # all other layers, hidden_dim => hidden_dim
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(utils.get_act_fn(act_fn))

        return layers


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu', shuffle_objects=False):
        super(EncoderCNNSmall, self).__init__()
        self.shuffle_objects = shuffle_objects
        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        if self.shuffle_objects:
            idx = torch.randperm(h.size(1))
            return h[:, idx]

        return h
    
    
class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu', shuffle_objects=False):
        super(EncoderCNNMedium, self).__init__()
        self.shuffle_objects = shuffle_objects

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        if self.shuffle_objects:
            idx = torch.randperm(h.size(1))
            return h[:, idx]

        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu', shuffle_objects=False):
        super(EncoderCNNLarge, self).__init__()
        self.shuffle_objects = shuffle_objects

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        h = self.act4(self.cnn4(h))
        if self.shuffle_objects:
            idx = torch.randperm(h.size(1))
            return h[:, idx]

        return h


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class DecoderMLP(nn.Module):
    """MLP decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.output_size = output_size

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        obj_ids = torch.arange(self.num_objects)
        obj_ids = utils.to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h = torch.cat((ins, obj_ids), -1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1],
                      self.output_size[2])


class DecoderCNNSmall(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNSmall, self).__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=10, stride=10)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.deconv1(h_conv))
        return self.deconv2(h)


class DecoderCNNMedium(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNMedium, self).__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=5, stride=5)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=9, padding=4)

        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        return self.deconv2(h)


class DecoderCNNLarge(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNLarge, self).__init__()

        width, height = output_size[1], output_size[2]

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=3, padding=1)

        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.ln2 = nn.BatchNorm2d(hidden_dim)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)
        self.act4 = utils.get_act_fn(act_fn)
        self.act5 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        h = self.act4(self.ln1(self.deconv2(h)))
        h = self.act5(self.ln1(self.deconv3(h)))
        return self.deconv4(h)


class AttentionV1(nn.Module):
    HIDDEN_SIZE = 512

    def __init__(self, state_size, action_size, key_query_size, value_size, sqrt_scale,
                 ablate_weights=False, use_sigmoid=False):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.key_query_size = key_query_size
        self.value_size = value_size
        self.sqrt_scale = sqrt_scale
        self.ablate_weights = ablate_weights
        self.use_sigmoid = use_sigmoid

        if self.use_sigmoid:
            self.normalizer = lambda x, dim: F.sigmoid(x)
        else:
            self.normalizer = F.softmax

        self.fc_key = MLP(self.state_size, self.key_query_size, self.HIDDEN_SIZE)
        self.fc_query = MLP(self.action_size, self.key_query_size, self.HIDDEN_SIZE)
        self.fc_value = MLP(self.action_size, self.value_size, self.HIDDEN_SIZE)

    def forward(self, x):
        state, action = x

        # flatten state
        batch_size = state.size(0)
        obj_size = state.size(1)
        state_r = state.reshape(batch_size * obj_size, state.size(2))

        # create keys and queries
        key_r = self.fc_key(state_r)
        query = self.fc_query(action)
        value = self.fc_value(action)

        key = key_r.reshape(batch_size, obj_size, self.key_query_size)

        # compute a vector of attention weights, one for each object slot
        if self.sqrt_scale:
            weights = self.normalizer((key * query[:, None]).sum(dim=2) * (1 / np.sqrt(self.key_query_size)), dim=-1)
        else:
            weights = self.normalizer((key * query[:, None]).sum(dim=2), dim=-1)

        if self.ablate_weights:
            # set uniform weights to check if they provide any benefit
            weights = torch.ones_like(weights) / weights.shape[1]

        # create a separate action for each object slot
        # weights: [|B|, |O|], value: [|B|, value_size]
        # => [|B|, |O|, value_size]
        return weights[:, :, None] * value[:, None, :]

    def forward_weights(self, x):
        state, action = x

        # flatten state
        batch_size = state.size(0)
        obj_size = state.size(1)
        state_r = state.reshape(batch_size * obj_size, state.size(2)) # 5000 x 2

        # create keys and queries
        key_r = self.fc_key(state_r) # 5000 x 512
        query = self.fc_query(action) # 1000 x 512

        key = key_r.reshape(batch_size, obj_size, self.key_query_size) # 1000 x 5 x 512

        # compute a vector of attention weights, one for each object slot
        if self.sqrt_scale:
            weights = self.normalizer((key * query[:, None]).sum(dim=2) * (1 / np.sqrt(self.key_query_size)), dim=-1) # 1000 x 5
        else:
            weights = self.normalizer((key * query[:, None]).sum(dim=2), dim=-1)

        if self.ablate_weights:
            # set uniform weights to check if they provide any benefit
            weights = torch.ones_like(weights) / weights.shape[1]

        return weights


class MLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, act_fn='relu'):
        super(MLP, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class ContrastiveSWMHA(ContrastiveSWM):
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim, num_objects, hinge=1., sigma=0.5,
                 encoder='large', ignore_action=False, copy_action=False, shuffle_objects=False, use_interactions=True,
                 num_layers=3, key_query_size=512, value_size=512):
        super().__init__(embedding_dim, input_dims, hidden_dim, action_dim, num_objects, hinge, sigma, encoder,
                         ignore_action, copy_action, shuffle_objects, use_interactions, num_layers)

        self.attention = AttentionV1(
            state_size=self.embedding_dim,
            action_size=self.action_dim,
            key_query_size=key_query_size,
            value_size=value_size,
            sqrt_scale=True,
            use_sigmoid=False
        )

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
            return norm * diff.pow(2).mean(2)
        else:
            pred_state, weights = self.forward_transition(state, action, all=True)
            diff = pred_state - next_state[:, None]
            diff = diff.pow(2).mean(dim=(2, 3))
            diff = (diff * weights).sum(1)
            return norm * diff

    def forward_transition(self, state, action, all=False):
        if len(action.shape) == 1:
            action = utils.to_one_hot(action, self.action_dim)
        else:
            assert len(action.shape) == 2
        weights = self.attention.forward_weights([state, action])

        if all:
            pred_state = []
            for obj_idx in range(self.num_objects):
                node_idx = torch.zeros(action.size(0), dtype=torch.long, device=action.device) + obj_idx
                tmp_action = self.action_to_target_node(action, node_idx)
                pred_trans, _, _ = self.transition_model([state, tmp_action, False])
                pred_state.append(state + pred_trans)
            return torch.stack(pred_state, dim=1), weights
        else:
            node_idx = torch.argmax(weights, dim=1)
            action = self.action_to_target_node(action, node_idx)
            pred_trans, _, _ = self.transition_model([state, action, False])
            return state + pred_trans

    def forward_weights(self, state, action):
        if len(action.shape) == 1:
            action = utils.to_one_hot(action, self.action_dim)
        else:
            assert len(action.shape) == 2
        return self.attention.forward_weights([state, action])

    def action_to_target_node(self, action, node_idx):
        new_action = torch.zeros(
            (action.size(0), self.num_objects, action.size(1)),
            dtype=torch.float32, device=action.device
        )
        indices = list(range(action.size(0)))
        new_action[indices, node_idx] = action.detach()
        return new_action


class ActionConverter:
    def __init__(self, action_dim: int, attention_module: AttentionV1 = None):
        super().__init__()
        self.action_dim = action_dim
        self.attention_module = attention_module

        assert self.attention_module is None or self.action_dim == self.attention_module.action_size

    def convert(self, state, action):
        num_objects = state.size()[1]
        if len(action.shape) == 1:
            action = utils.to_one_hot(action, self.attention_module.action_size)
        else:
            assert len(action.shape) == 2

        if self.attention_module is None:
            # copy action to all slots if action-attention is not used
            return action.unsqueeze(1).expand(state.size()[0], num_objects, self.action_dim)

        weights = self.attention_module.forward_weights([state, action])
        node_idx = torch.argmax(weights, dim=1)
        return self.action_to_target_node(action, node_idx, num_objects)

    def action_to_target_node(self, action, node_idx, num_objects):
        new_action = torch.zeros(
            (action.size(0), num_objects, action.size(1)),
            dtype=torch.float32, device=action.device
        )
        indices = list(range(action.size(0)))
        new_action[indices, node_idx] = action.detach()
        return new_action
