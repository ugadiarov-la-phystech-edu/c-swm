import collections
import random

import numpy as np


class PathFinder:
    def __init__(self, width, source, target, occupied_positions, is_x_first, debug=False):
        self.debug = debug
        self.is_x_first = is_x_first
        self.width = width
        self.source = source
        self.target = target
        self.occupied_positions = occupied_positions
        self.map = [[None] * self.width for _ in range(self.width)]
        self.visited = [[False] * self.width for _ in range(self.width)]
        self.visited[source[0]][source[1]] = True
        for position in self.occupied_positions:
            self.visited[position[0]][position[1]] = True

        self.visited[target[0]][target[1]] = False

        self.queue = collections.deque()
        self.queue.append(self.source)
        if self.debug:
            print('MAP')
            for vector in self.visited:
                print(['x' if v else 'o' for v in vector])

    def find_path(self):
        does_path_exist = self._find_path()
        if not does_path_exist:
            return None

        positions = [self.target]
        position = positions[0]
        while position != self.source:
            position = self.map[position[0]][position[1]]
            positions.append(position)

        if self.debug:
            print('REACHABILITY')
            for vector in self.map:
                print(vector)

        return positions[::-1]

    def _neighbours(self, position):
        neighbours = []
        if position[0] > 0:
            neighbours.append((position[0] - 1, position[1]))
        if position[0] < self.width - 1:
            neighbours.append((position[0] + 1, position[1]))
        if position[1] > 0:
            neighbours.append((position[0], position[1] - 1))
        if position[1] < self.width - 1:
            neighbours.append((position[0], position[1] + 1))

        if self.is_x_first:
            return neighbours

        return neighbours[::-1]

    def _find_path(self):
        while len(self.queue) > 0:
            current_position = self.queue.popleft()
            neighbours = self._neighbours(current_position)
            for neighbour in neighbours:
                if not self.visited[neighbour[0]][neighbour[1]]:
                    self.map[neighbour[0]][neighbour[1]] = current_position
                    self.visited[neighbour[0]][neighbour[1]] = True
                    self.queue.append(neighbour)

                if neighbour == self.target:
                    return True

        return False


if __name__ == '__main__':
    width = 5
    source = (0, 0)
    target = (4, 4)
    occupied = []
    locations = np.random.choice(width * width, 8, replace=False)
    for x, y in zip(*np.unravel_index(locations, [width, width])):
        position = (x, y)
        if position not in (source, target):
            occupied.append(position)

    finder = PathFinder(width, source, target, occupied, debug=True)
    path = finder.find_path()
    print(path)
