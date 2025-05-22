from typing import Self, TypeVar, List

import numpy as np

from .obstacles import DinCircles, LetterC

class Area:
    """
    Class of the 2D area containing obstacles
    """
    def __init__(self):
        self.obstacles = None

    def random_static_set(self, area, n_obj: int, min_size: float, max_size: float, excluded_area: List or None = None,
                          _type: str = 'static', frequency: float = 50) -> Self:
        """
        В процессе.
        Заполняется пока только кругами
        Returns
        -------

        """
        if _type == 'static':
            self.obstacles = DinCircles.get_random(area, min_size, max_size, excluded_area=excluded_area, n=n_obj,
                                                   frequency=frequency)
            self.obstacles.static = True
        if _type == 'dynamic':
            self.obstacles = DinCircles.get_random(area, min_size, max_size, excluded_area=excluded_area, n=n_obj,
                                                   frequency=frequency)
        if _type == 'letter':
            self.obstacles = LetterC.get_random(area, min_size, max_size, n=n_obj, frequency=frequency,
                                                excluded_area=excluded_area)

        return self

    def add(self, obs) -> Self:
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
        self.obstacles.update()

    def parse(self):
        """
        In progress
        Returns
        -------

        """
        dt_circle = np.dtype([('size', float), ('coordinates', float, (2, ))])
        static_circle = []
        for i in range(len(self.obstacles.x)):
            x = self.obstacles.x[i]
            size = self.obstacles.size[i]
            static_circle.append((size, x))
        static_circle = np.array(static_circle, dtype=dt_circle)
        return static_circle
