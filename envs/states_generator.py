import collections

import gym
import numpy as np

from matplotlib import pyplot as plt

import envs


class StatesGenerator:
    def __init__(self, seed, env):
        self.env = env
        self.seed = seed
        self.np_random = np.random.RandomState(self.seed)

    @staticmethod
    def is_around(position, center, distance):
        return max(abs(position[0] - center[0]), abs(position[1] - center[1])) <= distance

    def sample_position(self, box_id):
        while True:
            if box_id == 0:
                locs = self.np_random.choice(self.env.w ** 2, 1, replace=False)
                xs, ys = np.unravel_index(locs, [self.env.w, self.env.w])
            else:
                locs = self.np_random.choice((self.env.w - 2) ** 2, 1, replace=False)
                xs, ys = np.unravel_index(locs, [self.env.w - 2, self.env.w - 2])
                xs += 1
                ys += 1

            x = xs[0]
            y = ys[0]

            if self.env.state[x, y] == -1:
                return x, y

    def sample_position_straight(self, position, box_id, distance):
        shifts = [(-distance, 0), (0, -distance), (0, distance), (distance, 0)]

        for _ in range(len(shifts)):
            shift = shifts.pop(self.np_random.randint(len(shifts)))
            x = position[0] + shift[0]
            y = position[1] + shift[1]
            if self.env._is_in_grid((x, y), box_id) and self.env.state[x, y] == -1:
                return x, y, tuple(-coordinate for coordinate in shift)

        assert False, 'Cannot be here!'

    def sample_position_around(self, position, box_id, distance):
        side = 2 * distance + 1
        while True:
            locs = self.np_random.choice(side ** 2, 1, replace=False)
            x, y = [coordinates[0] for coordinates in np.unravel_index(locs, [side, side])]
            if x == distance and y == distance:
                continue

            x += position[0] - distance
            y += position[1] - distance
            if self.env._is_in_grid((x, y), box_id) and self.env.state[x, y] == -1:
                return x, y

    def sample_positions_along_perimeter(self, width):
        loc = self.np_random.choice(4 * (width - 1))
        side = loc // (width - 1)
        shift = loc % (width - 1)
        if side == 0:
            x = 0
            y = shift
        elif side == 1:
            x = shift
            y = width - 1
        elif side == 2:
            x = width - 1
            y = width - 1 - shift
        elif side == 3:
            x = width - 1 - shift
            y = 0
        else:
            assert False, f'Cannot be here!'

        vectors = []
        if x == 0:
            vectors.append((-1, 0))
        if y == width - 1:
            vectors.append((0, 1))
        if x == width - 1:
            vectors.append((1, 0))
        if y == 0:
            vectors.append((0, -1))

        idx = self.np_random.choice(a=len(vectors))
        return x, y, vectors[idx]
    
    def sample_position_border(self):
        while True:
            x, y, vector = self.sample_positions_along_perimeter(self.env.w)
            if self.env.state[x, y] == -1:
                return x, y, vector

    def sample_position_next_to_border(self):
        while True:
            x, y, vector = self.sample_positions_along_perimeter(self.env.w - 2)
            if self.env.state[x + 1, y + 1] == -1:
                return x + 1, y + 1, vector

    def agent_box_goal(self):
        self.env._clear_state()

        goal_id = next(iter(self.env.goal_ids))
        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        box_id = self.np_random.randint(low=1, high=goal_id)
        box_x, box_y = self.sample_position_around((goal_x, goal_y), box_id, distance=2)
        self.env._set_position(box_id, box_x, box_y)

        agent_id = 0
        agent_x, agent_y = self.sample_position_around((box_x, box_y), agent_id, distance=1)
        self.env._set_position(agent_id, agent_x, agent_y)

        remaining_box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id, box_id})
        remaining_box_ids = self.np_random.choice(
            remaining_box_ids, size=self.np_random.randint(low=0, high=len(remaining_box_ids) + 1), replace=False
        )
        for another_box_id in remaining_box_ids:
            another_box_x, another_box_y = self.sample_position(another_box_id)
            self.env._set_position(another_box_id, another_box_x, another_box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = remaining_box_ids.shape[0] + 1

    def agent_box_box(self):
        self.env._clear_state()

        goal_id = next(iter(self.env.goal_ids))

        box1_id, box2_id = self.np_random.choice(np.arange(start=1, stop=goal_id), size=2, replace=False)
        box1_x, box1_y = self.sample_position(box1_id)
        self.env._set_position(box1_id, box1_x, box1_y)

        box2_x, box2_y = self.sample_position_around((box1_x, box1_y), box2_id, distance=2)
        self.env._set_position(box2_id, box2_x, box2_y)

        agent_id = 0
        agent_x, agent_y = self.sample_position_around((box1_x, box1_y), agent_id, distance=1)
        self.env._set_position(agent_id, agent_x, agent_y)

        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        remaining_box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id, box1_id, box2_id})
        remaining_box_ids = self.np_random.choice(
            remaining_box_ids, size=self.np_random.randint(low=0, high=len(remaining_box_ids) + 1), replace=False
        )
        for another_box_id in remaining_box_ids:
            another_box_x, another_box_y = self.sample_position(another_box_id)
            self.env._set_position(another_box_id, another_box_x, another_box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = remaining_box_ids.shape[0] + 2

    def agent_goal(self):
        self.env._clear_state()

        goal_id = next(iter(self.env.goal_ids))
        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        agent_id = 0
        agent_x, agent_y, vector = self.sample_position_straight((goal_x, goal_y), agent_id, distance=1)
        self.env._set_position(agent_id, agent_x, agent_y)

        remaining_box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id})
        remaining_box_ids = self.np_random.choice(
            remaining_box_ids, size=self.np_random.randint(low=1, high=len(remaining_box_ids) + 1), replace=False
        )
        for another_box_id in remaining_box_ids:
            another_box_x, another_box_y = self.sample_position(another_box_id)
            self.env._set_position(another_box_id, another_box_x, another_box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = remaining_box_ids.shape[0]

        return vector

    def agent_border(self):
        self.env._clear_state()

        agent_id = 0
        agent_x, agent_y, vector = self.sample_position_border()
        self.env._set_position(agent_id, agent_x, agent_y)

        goal_id = next(iter(self.env.goal_ids))
        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        remaining_box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id})
        remaining_box_ids = self.np_random.choice(
            remaining_box_ids, size=self.np_random.randint(low=1, high=len(remaining_box_ids) + 1), replace=False
        )
        for another_box_id in remaining_box_ids:
            another_box_x, another_box_y = self.sample_position(another_box_id)
            self.env._set_position(another_box_id, another_box_x, another_box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = remaining_box_ids.shape[0]
        return vector

    def agent_box_border(self):
        self.env._clear_state()

        agent_id = 0
        goal_id = next(iter(self.env.goal_ids))
        box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id})
        box_id = self.np_random.choice(box_ids)

        box_x, box_y, vector = self.sample_position_next_to_border()
        self.env._set_position(box_id, box_x, box_y)
        assert 1 <= box_x <= self.env.w - 1, f'{box_x}'
        assert 1 <= box_y <= self.env.w - 1, f'{box_y}'

        agent_x, agent_y = box_x - vector[0], box_y - vector[1]
        self.env._set_position(agent_id, agent_x, agent_y)
        assert 1 <= agent_x <= self.env.w - 1, f'{agent_x}'
        assert 1 <= agent_y <= self.env.w - 1, f'{agent_y}'

        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        remaining_box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id, box_id})
        remaining_box_ids = self.np_random.choice(
            remaining_box_ids, size=self.np_random.randint(low=0, high=len(remaining_box_ids) + 1), replace=False
        )
        for another_box_id in remaining_box_ids:
            another_box_x, another_box_y = self.sample_position(another_box_id)
            self.env._set_position(another_box_id, another_box_x, another_box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = remaining_box_ids.shape[0] + 1
        return vector

    def agent_box_box_stack(self):
        self.env._clear_state()

        agent_id = 0
        goal_id = next(iter(self.env.goal_ids))
        box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id})

        box1_id, box2_id = self.np_random.choice(box_ids, size=2, replace=False)

        box1_x, box1_y = self.sample_position(box1_id)
        self.env._set_position(box1_id, box1_x, box1_y)

        box2_x, box2_y, vector = self.sample_position_straight((box1_x, box1_y), box2_id, distance=1)
        self.env._set_position(box2_id, box2_x, box2_y)

        agent_x, agent_y = box2_x - vector[0], box2_y - vector[1]
        assert 0 <= agent_x < self.env.w, f'{agent_x}'
        assert 0 <= agent_y < self.env.w, f'{agent_y}'
        self.env._set_position(agent_id, agent_x, agent_y)

        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        remaining_box_ids = list(set(range(self.env.n_boxes)) - {goal_id, agent_id, box1_id, box2_id})
        remaining_box_ids = self.np_random.choice(
            remaining_box_ids, size=self.np_random.randint(low=0, high=len(remaining_box_ids) + 1), replace=False
        )
        for another_box_id in remaining_box_ids:
            another_box_x, another_box_y = self.sample_position(another_box_id)
            self.env._set_position(another_box_id, another_box_x, another_box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = remaining_box_ids.shape[0] + 2

        return vector

    def agent_box_around_goal(self):
        self.env._clear_state()
        goal_id = next(iter(self.env.goal_ids))
        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        agent_id = 0
        agent_x, agent_y = self.sample_position_around((goal_x, goal_y), agent_id, distance=1)
        self.env._set_position(agent_id, agent_x, agent_y)

        box_around_id = self.np_random.choice([1, 2, 3])
        box_around_x, box_around_y = self.sample_position_around((goal_x, goal_y), box_around_id, distance=1)
        self.env._set_position(box_around_id, box_around_x, box_around_y)

        for box_id in set(range(self.env.n_boxes)) - {goal_id, agent_id, box_around_id}:
            box_x, box_y = self.sample_position(box_id)
            self.env._set_position(box_id, box_x, box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = 3

    def agent_box_around_goal_side_by_side(self):
        self.env._clear_state()
        goal_id = next(iter(self.env.goal_ids))
        goal_x, goal_y = self.sample_position(goal_id)
        self.env._set_position(goal_id, goal_x, goal_y)

        close_box_id = self.np_random.choice([1, 2, 3])
        close_box_x, close_box_y = self.sample_position_around((goal_x, goal_y), close_box_id, distance=1)
        self.env._set_position(close_box_id, close_box_x, close_box_y)

        agent_id = 0
        while True:
            agent_x, agent_y, vector = self.sample_position_straight((close_box_x, close_box_y), agent_id, distance=1)
            if StatesGenerator.is_around((agent_x, agent_y), (goal_x, goal_y), distance=1):
                break

        self.env._set_position(agent_id, agent_x, agent_y)

        for box_id in set(range(self.env.n_boxes)) - {goal_id, agent_id, close_box_id}:
            box_x, box_y = self.sample_position(box_id)
            self.env._set_position(box_id, box_x, box_y)

        self.env.steps_taken = 0
        self.env.n_boxes_in_game = 3

    def agent_box_near_goal_stack(self):
        pass

    def agent_box_near_goal_reverse_stack(self):
        pass

    def agent_two_boxes(self):
        pass


if __name__ == '__main__':
    seed = 8
    env = gym.make('ShapesChannelWiseTernaryInteractionsBoxes5Width7EmbodiedAgentOneStaticGoalTrain-v0')
    env.seed(seed)
    env.reset()
    generator = StatesGenerator(seed, env.unwrapped)
    vector = generator.agent_box_box_stack()
    print(vector)
    # counter = collections.Counter()
    # for _ in range(1000):
    #     generator.agent_box_goal()
    #     counter[env.n_boxes_in_game] += 1
    #
    # print(counter)

    image = np.repeat(env.unwrapped._get_observation()[1].sum(axis=0)[:, :, np.newaxis], repeats=3, axis=2)
    plt.imshow(image)
    plt.show()

    action = env.direction2action[vector]
    o, r, d, i = env.step(action)
    print(f'reward={r} done={d} info={i}')
    image = np.repeat(o[1].sum(axis=0)[:, :, np.newaxis], repeats=3, axis=2)
    plt.imshow(image)
    plt.show()

