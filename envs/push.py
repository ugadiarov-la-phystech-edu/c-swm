import gym
import numpy as np
import skimage
from gym.utils import seeding
from gym import spaces
#from scipy.misc import imresize
import sys
import copy

from matplotlib import pyplot as plt

import utils


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width // 2], [c0 + width // 2, c0, c0 + width // 2, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def triangle(r0, c0, width, im_size):
    rr = np.asarray([r0 + 4] * 5 + [r0 + 3] * 3 + [r0 + 2] * 3 + [r0 + 1] + [r0], dtype=np.int32)
    cc = np.asarray(list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) * 2 + [c0 + 2] * 2, dtype=np.int32)
    # rr, cc = [r0, r0 + width - 1, r0 + width - 1], [c0 + width // 2, c0, c0 + width - 1]
    # return skimage.draw.polygon(rr, cc, im_size)
    return rr, cc

def circle(r0, c0, width, im_size):
    rr = np.asarray([r0] + [r0 + 1] * 3 + [r0 + 2] * 5 + [r0 + 3] * 3 + [r0 + 4], dtype=np.int32)
    cc = np.asarray([c0 + 2] + list(range(c0 + 1, c0 + 4)) + list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) + [c0 + 2], dtype=np.int32)
    # rr, cc = [r0, r0 + width - 1, r0 + width - 1], [c0 + width // 2, c0, c0 + width - 1]
    # return skimage.draw.polygon(rr, cc, im_size)
    return rr, cc

def cross(r0, c0, width, im_size):
    diff1 = width // 3 + 1
    diff2 = 2 * width // 3
    rr = [r0 + diff1, r0 + diff2, r0 + diff2, r0 + width, r0 + width,
            r0 + diff2, r0 + diff2, r0 + diff1, r0 + diff1, r0, r0, r0 + diff1]
    cc = [c0, c0, c0 + diff1, c0 + diff1, c0 + diff2, c0 + diff2, c0 + width,
            c0 + width, c0 + diff2, c0 + diff2, c0 + diff1, c0 + diff1]
    return skimage.draw.polygon(rr, cc, im_size)

def pentagon(r0, c0, width, im_size):
    diff1 = width // 3 - 1
    diff2 = 2 * width // 3 + 1
    rr = [r0 + width // 2, r0 + width, r0 + width, r0 + width // 2, r0]
    cc = [c0, c0 + diff1, c0 + diff2, c0 + width, c0 + width // 2]
    return skimage.draw.polygon(rr, cc, im_size)

def parallelogram(r0, c0, width, im_size):
    rr = np.asarray([r0] * 2 + [r0 + 1] * 3 + [r0 + 2] * 3 + [r0 + 3] * 3 + [r0 + 4] * 2, dtype=np.int32)
    cc = np.asarray([c0, c0 + 1] + list(range(c0, c0 + 3)) + list(range(c0 + 1, c0 + 4)) + list(range(c0 + 2, c0 + 5)) + list(range(c0 + 3, c0 + 5)), dtype=np.int32)
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0 + width // 2, c0 + width, c0 + width - width // 2]
    # return skimage.draw.polygon(rr, cc, im_size)
    return rr, cc


def scalene_triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width//2], [c0 + width - width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


class Push(gym.Env):
    def __init__(self, mode='default', n_boxes=4, n_goals=2, observation_type='squares', max_episode_steps=75, hard_walls=False, seed=None):
        self.w = 10
        self.step_limit = max_episode_steps
        self.n_boxes = n_boxes
        self.n_obstacles = 0
        self.n_goals = n_goals
        self.box_block = True
        self.walls = False
        self.soft_obstacles = True
        self.render_scale = 5
        self.hard_walls = hard_walls
        self.colors = utils.get_colors(num_colors=max(9, self.n_boxes + self.n_goals))
        self.observation_type = observation_type

        self.use_obst = self.n_obstacles > 0 or self.walls

        self.directions = {
            0: np.asarray((1, 0)),
            1: np.asarray((0, -1)),
            2: np.asarray((-1, 0)),
            3: np.asarray((0, 1))
        }

        self.np_random = None
        self.channels_first = True

        self.action_space = spaces.Discrete(4 * self.n_boxes)
        if self.observation_type == 'grid':
            self.observation_space = spaces.Box(
                0,
                1,
                (self.w, self.w, self.n_goals + self.n_boxes + self.n_obstacles)
            )
            # channels are n_goals, n_boxes, n_obstacles, time remaining
        elif self.observation_type in ('squares', 'shapes'):
            self.observation_space = spaces.Box(0, 255, (3, self.w * self.render_scale, self.w * self.render_scale), dtype=np.uint8)
        else:
            raise ValueError(f'Invalid observation_type: {self.observation_type}.')

        self.state = None
        self.steps_taken = 0
        self.pos = None
        self.image = None
        self.box_pos = np.zeros(shape=(self.n_boxes, 2), dtype=np.int32)
        self.goal_pos = np.zeros(shape=(self.n_goals, 2), dtype=np.int32)

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        if self.observation_type == 'squares':
            image = self.render_squares()
        elif self.observation_type == 'shapes':
            image = self.render_shapes()
        else:
            assert False, f'Invalid observation type: {self.observation_type}.'

        return self.state, image

    def reset(self):
        state = np.zeros([self.w, self.w, self.n_goals + self.n_boxes + self.n_obstacles + 1])
        # initialise time-remaining to 1.
        state[:, :, self.n_goals + self.n_boxes + self.n_obstacles].fill(1)

        # fill in walls at borders
        if self.walls:
            assert False, 'Walls are not supported in channel-wise version'
            state[0, :, 2].fill(1)
            state[self.w - 1, :, 2].fill(1)
            state[:, 0, 2].fill(1)
            state[:, self.w - 1, 2].fill(1)

        # sample random locations for self, goals, and walls.
        locs = np.random.choice((self.w - 2 - (2 * self.walls)) ** 2, self.n_goals + self.n_boxes + self.n_obstacles, replace=False)

        xs, ys = np.unravel_index(locs, [self.w - 2 - (2 * self.walls), self.w - 2 - (2 * self.walls)])
        xs += 1 + self.walls
        ys += 1 + self.walls

        # populate state with locations
        for i, (x, y) in enumerate(zip(xs, ys)):
            state[x, y, i] = 1
            if i < self.n_goals:
                goal_id = i
                self.goal_pos[goal_id, :] = x, y
            elif self.n_goals <= i < self.n_goals + self.n_boxes:
                box_id = i - self.n_goals
                self.box_pos[box_id, :] = x, y
            else:
                assert self.n_obstacles > 0

        self.state = state
        self.steps_taken = 0
        self.boxes_left = self.n_boxes
        self.edge_boxes = 0

        return self._get_observation()

    def _box_channel(self, box_id):
        return self.n_goals + box_id

    def _current_map(self):
        # channels are 0: goals, 1: boxes, 2: obstacles
        current_map = np.zeros((self.w, self.w, 2 + self.use_obst))
        current_map[:, :, 0] = self.state[:, :, :self.n_goals].sum(axis=-1)
        current_map[:, :, 1] = self.state[:, :, self.n_goals:self.n_goals + self.n_boxes].sum(axis=-1)

        if self.use_obst:
            current_map[:, :, 2] = self.state[:, :, self.n_goals + self.n_boxes:].sum(axis=-1)

        if not self.soft_obstacles:
            assert np.max(current_map) <= 1

        return current_map

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        direction = action % 4
        vec = self.directions[direction]

        box_id = action // 4
        old_box_pos = self.box_pos[box_id]
        box_pos = old_box_pos + vec
        box_channel = self._box_channel(box_id)

        done = False
        reward = -0.01
        current_map = self._current_map()

        if self.state[old_box_pos[0], old_box_pos[1], box_channel] == 0:
            # This box is out of the game. There is nothing to do.
            pass
        elif not self.is_in_grid(box_pos):
            if not self.hard_walls:
                # push out of grid, destroy
                self.state[old_box_pos[0], old_box_pos[1], box_channel] = 0
                self.box_pos[box_id] = -1, -1
                self.boxes_left -= 1
                reward += -0.1
        elif current_map[box_pos[0], box_pos[1], 1] == 1:
            # push into another box,
            if self.box_block:
                # blocking, so no movement
                reward += -0.1
            else:
                # not blocking, so destroy
                self.state[old_box_pos[0], old_box_pos[1], box_channel] = 0
                self.box_pos[box_id] = -1, -1
                self.boxes_left -= 1
                reward += -0.1
        elif self.use_obst and current_map[box_pos[0], box_pos[1], 2] == 1:
            # push into obstacle
            if self.soft_obstacles:
                reward += -0.2
                self.state[old_box_pos[0], old_box_pos[1], box_channel] = 0
                self.state[box_pos[0], box_pos[1], box_channel] = 1
                self.box_pos[box_id] = box_pos
            else:
                reward += -0.1
        elif current_map[box_pos[0], box_pos[1], 0] == 1:
            # pushed into goal, get reward
            self.boxes_left -= 1
            self.state[old_box_pos[0], old_box_pos[1], box_channel] = 0
            self.box_pos[box_id] = -1, -1
            reward += 1.0
        else:
            # pushed into open space, move box
            self.state[old_box_pos[0], old_box_pos[1], box_channel] = 0
            self.state[box_pos[0], box_pos[1], box_channel] = 1
            self.box_pos[box_id] = box_pos

        self.steps_taken += 1
        if self.steps_taken >= self.step_limit:
            done = True

        if self.boxes_left == 0:
            done = True

        return self._get_observation(), reward, done, {}

    def is_in_grid(self, point):
        return (0 <= point[0] < self.w) and (0 <= point[1] < self.w)

    def is_on_edge(self, point):
        return (0 == point[0]) or (0 == point[1]) or (self.w - 1 == point[0]) or (self.w - 1 == point[1])

    def print(self, message=''):
        out = np.zeros([self.w, self.w])
        state = self._current_map()
        if self.use_obst:
            out[state[:, :, 2].astype(bool)] = 4
        out[state[:, :, 1].astype(bool)] = 3
        out[state[:, :, 0].astype(bool)] = 2
        chars = {0: ".", 1: "x", 2: "O", 3: "#", 4:"@"}
        pretty = "\n".join(["".join([chars[x] for x in row]) for row in out])
        print(pretty)
        print("TIMELEFT ", self.step_limit - self.steps_taken, message)

    def clone_full_state(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def restore_full_state(self, state_dict):
        self.__dict__.update(state_dict)

    def get_action_meanings(self):
        return ["down", "left", "up", "right"] * self.n_boxes

    def render_squares(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, 3), dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            rr, cc = square(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            im[rr, cc, :] = self.colors[idx][:3]

        for idx, pos in enumerate(self.goal_pos):
            rr, cc = square(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            im[rr, cc, :] = 1

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255

        return im.astype(dtype=np.uint8)

    def render_shapes(self,):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, 3), dtype=np.float32)
        for idx, pos in enumerate(np.concatenate([self.box_pos, self.goal_pos], axis=0)):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            shape_id = idx % 8
            if shape_id == 0:
                # radius = self.render_scale // 2
                # rr, cc = skimage.draw.circle(
                #     pos[0] * self.render_scale + radius, pos[1] * self.render_scale + radius, radius, im.shape)
                rr, cc = circle(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 1:
                rr, cc = triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 2:
                rr, cc = square(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 3:
                rr, cc = parallelogram(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 4:
                rr, cc = cross(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 5:
                rr, cc = diamond(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 6:
                rr, cc = pentagon(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            else:
                rr, cc = scalene_triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)

            im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255
        return im.astype(np.uint8)


if __name__ == "__main__":
    """
    If called directly with argument "random", evaluates the average return of a random policy.
    If called without arguments, starts an interactive game played with wasd to move, q to quit.
    """
    env = Push()

    if len(sys.argv) > 1 and sys.argv[1] == "random":
        all_r = []
        n_episodes = 1000
        for i in range(n_episodes):
            s = env.reset()
            plt.imshow(s[1])
            plt.show()
            done = False
            episode_r =0
            while not done:
                s, r, done, _ = env.step(np.random.randint(env.action_space.n))
                episode_r += r
                plt.imshow(s[1])
                plt.show()
            all_r.append(episode_r)
        print(np.mean(all_r), np.std(all_r), np.std(all_r)/np.sqrt(n_episodes))
    else:
        s = env.reset()
        plt.imshow(s[1])
        plt.show()
        env.print()
        episode_r = 0

        while True:
            keys = list(input())
            if keys[0] == "q":
                break

            obj_id = int(keys[0])
            key = keys[1]
            if key == "a":
                a = 1
            elif key == "s":
                a = 0
            elif key == "d":
                a = 3
            elif key == "w":
                a = 2
            else:
                raise ValueError('Invalid action key:', key)

            a += 4 * obj_id

            s, r, d, _ = env.step(a)
            episode_r += r
            env.print(f'. Reward: {episode_r}')
            plt.imshow(env.render_squares().transpose([1, 2, 0]))
            plt.show()
            if d or key == "r":
                print("Done with {} points. Resetting!".format(episode_r))
                s = env.reset()
                episode_r = 0
                env.print()