"""Gym environment for block pushing tasks (2D Shapes and 3D Cubes)."""

import numpy as np

import utils
import gym
from gym import spaces
from gym.utils import seeding

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


import skimage


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width//2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def parallelogram(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0 + width // 2, c0 + width, c0 + width - width // 2]
    return skimage.draw.polygon(rr, cc, im_size)


def cross(r0, c0, width, im_size):
    diff1 = width // 3 + 1
    diff2 = 2 * width // 3
    rr = [r0 + diff1, r0 + diff2, r0 + diff2, r0 + width, r0 + width,
            r0 + diff2, r0 + diff2, r0 + diff1, r0 + diff1, r0, r0, r0 + diff1]
    cc = [c0, c0, c0 + diff1, c0 + diff1, c0 + diff2, c0 + diff2, c0 + width,
            c0 + width, c0 + diff2, c0 + diff2, c0 + diff1, c0 + diff1]
    return skimage.draw.polygon(rr, cc, im_size)


def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(positions, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ['purple', 'green', 'orange', 'blue', 'brown']

    for i, pos in enumerate(positions):
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(  # Crop and resize
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS))
    return im / 255.


class BlockPushing(gym.Env):
    """Gym environment for block pushing task."""

    def __init__(self, width=5, height=5, render_type='shapes', num_objects=5, num_static_objects=0, num_goals=0, hard_walls=True,
                 seed=None, random_actions=False):
        self.width = width
        self.height = height
        self.render_type = render_type
        self.hard_walls = hard_walls

        self.num_objects = num_objects
        self.num_static_objects = num_static_objects
        self.num_goals = num_goals
        assert self.num_objects >= self.num_static_objects
        assert self.num_static_objects >= self.num_goals
        self.num_actions = 4 * (self.num_objects - self.num_static_objects)  # Move NESW

        self.colors = utils.get_colors(num_colors=max(9, self.num_objects))

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = None
        self.active_objects = None

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True
        self.random_actions = random_actions

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.width, self.height),
            dtype=np.uint8
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        im = None
        if self.render_type == 'grid':
            im = np.zeros((3, self.width, self.height))
            for idx, pos in enumerate(self.objects):
                if self.active_objects[idx]:
                    im[:, pos[0], pos[1]] = self.colors[idx][:3]
        elif self.render_type == 'circles':
            im = np.zeros((self.width*10, self.height*10, 3), dtype=np.float32)
            for idx, pos in enumerate(self.objects):
                if self.active_objects[idx]:
                    rr, cc = skimage.draw.circle(
                        pos[0]*10 + 5, pos[1]*10 + 5, 5, im.shape)
                    im[rr, cc, :] = self.colors[idx][:3]
        elif self.render_type == 'shapes':
            im = np.zeros((self.width*10, self.height*10, 3), dtype=np.float32)
            for idx, pos in enumerate(self.objects):
                shape_id = idx % 5
                if self.active_objects[idx]:
                    if shape_id == 0:
                        rr, cc = skimage.draw.circle(
                            pos[0]*10 + 5, pos[1]*10 + 5, 5, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    elif shape_id == 1:
                        rr, cc = triangle(
                            pos[0]*10, pos[1]*10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    elif shape_id == 2:
                        rr, cc = square(
                            pos[0]*10, pos[1]*10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    elif shape_id == 3:
                        rr, cc = parallelogram(
                            pos[0] * 10, pos[1] * 10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    else:
                        rr, cc = cross(
                            pos[0] * 10, pos[1] * 10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
        elif self.render_type == 'squares':
            im = np.zeros((self.width*10, self.height*10, 3), dtype=np.float32)
            for idx, pos in enumerate(self.objects):
                if self.active_objects[idx]:
                    rr, cc = square(
                        pos[0] * 10, pos[1] * 10, 10, im.shape)
                    im[rr, cc, :] = self.colors[idx][:3]
        elif self.render_type == 'cubes':
            im = render_cubes([pos for obj_id, pos in enumerate(self.objects) if self.active_objects[obj_id]], self.width)

        if self.render_type != 'grid':
            im = im.transpose([2, 0, 1])

        im *= 255
        return im.astype(np.uint8)

    def get_state(self):
        im = np.zeros(
            (self.num_objects, self.width, self.height), dtype=np.int32)
        for idx, pos in enumerate(self.objects):
            if self.active_objects[idx]:
                im[idx, pos[0], pos[1]] = 1
        return im

    def reset(self):
        self.objects = [[-1, -1] for _ in range(self.num_objects)]
        self.active_objects = [True] * self.num_objects

        # Randomize object position.
        for i in range(len(self.objects)):

            # Resample to ensure objects don't fall on same spot.
            while not self.is_in_grid(self.objects[i], i) or self.check_collisions(self.objects[i], i) is not None:
                self.objects[i] = [
                    np.random.choice(np.arange(self.width)),
                    np.random.choice(np.arange(self.height))
                ]

        state_obs = (self.get_state(), self.render())
        return state_obs

    def is_in_grid(self, pos, obj_id):
        """Check if position is in grid."""
        if pos[0] < 0 or pos[0] >= self.width:
            return False
        if pos[1] < 0 or pos[1] >= self.height:
            return False

        return True

    def check_collisions(self, pos, obj_id):
        """Check collisions."""
        if self.collisions:
            for idx, obj_pos in enumerate(self.objects):
                if not self.active_objects[idx] or idx == obj_id:
                    continue

                if pos[0] == obj_pos[0] and pos[1] == obj_pos[1]:
                    return idx

        return None

    def valid_move(self, obj_id, offset):
        """Check if move is valid."""
        old_pos = self.objects[obj_id]
        new_pos = [p + o for p, o in zip(old_pos, offset)]
        return self.check_collisions(new_pos, obj_id) is None

    def translate(self, obj_id, offset):
        """"Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) tuple of offsets.
        """
        old_pos = self.objects[obj_id]
        new_pos = [p + o for p, o in zip(old_pos, offset)]
        collision_obj_id = self.check_collisions(new_pos, obj_id)
        if collision_obj_id is None:
            if self.is_in_grid(new_pos, obj_id):
                self.objects[obj_id] = new_pos
            elif not self.hard_walls:
                self.active_objects[obj_id] = False
                self.objects[obj_id] = [-1, -1]
        elif collision_obj_id >= self.num_objects - self.num_goals:
            self.active_objects[obj_id] = False
            self.objects[obj_id] = [-1, -1]

    def step(self, action):

        done = False
        reward = 0

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        if self.random_actions:
            action = self.action_space.sample()

        direction = action % 4
        obj = action // 4

        if self.active_objects[obj]:
            self.translate(obj, directions[direction])

        state_obs = (self.get_state(), self.render())
        done = sum(self.active_objects) <= self.num_static_objects

        return state_obs, reward, done, {"TimeLimit.truncated": False}
