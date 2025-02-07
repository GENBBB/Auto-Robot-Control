from typing import Self, TypeVar

import numpy as np

from config import config
from .obstacles import Obstacle, Circle

n_obj = int(config['Map']['n_obj'])
min_size = float(config['Map']['min_size'])
max_size = float(config['Map']['max_size'])
width = float(config['Config']['width'])
height = float(config['Config']['height'])

obstacles_like = TypeVar('obstacles_like', bound=Circle)


class Area:
    """
    Class of the 2D area containing obstacles
    """
    def __init__(self, obstacles: list[obstacles_like] = None):
        """
        Parameters
        ----------
        obstacles: list of Obstacle
            list of obstacles on the map at the initial time
        """
        if obstacles is None:
            obstacles = []
        self.obstacles = obstacles

    def random_static_set(self) -> Self:
        """
        В процессе.
        Заполняется пока только кругами
        Returns
        -------

        """
        self.obstacles = [Circle.get_random(min_size, max_size, (0, width), (0, height)) for i in range(n_obj)]
        return self

    def add(self, obs: Obstacle or list[Obstacle]) -> Self:
        """
        Add obstacles to the map at the initial moment of time.
        When added at an arbitrary time other than zero, the behavior is undefined
        Parameters
        ----------
        obs: Obstacle or list[Obstacle]
            Obstacle or list of obstacles added to the area
        Returns
        -------
        self: Area
            With added obstacles
        """
        self.obstacles.append(obs)
        return self

    def update(self):
        """
        In progress
        Returns
        -------

        """
        pass

    def parse(self):
        """
        In progress
        Returns
        -------

        """
        dt_circle = np.dtype([('size', float), ('coordinates', float, (2, ))])
        static_circle = []
        for obj in self.obstacles:
            if type(obj) is Circle:
                static_circle.append((obj.size, obj.point))
        static_circle = np.array(static_circle, dtype=dt_circle)
        return static_circle
