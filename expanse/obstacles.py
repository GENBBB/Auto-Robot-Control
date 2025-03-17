from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import Self, List

import numpy as np
import skspatial.objects as sks_obj
from skspatial.objects import Point


class Obstacle(ABC):
    """
    Abstract 2D obstacle class
    """
    def update(self) -> Self:
        """
        Updated obstacle placement for dynamic obstacles.
        For non-dynamic obstacles, does not change the location
        For dynamic objects, also saves the trajectory in the corresponding attributes

        Returns
        -------
        self: Obstacle
            Updated obstacle
        """
        return self

    @abstractmethod
    def intersection(self, line: sks_obj.Line, point: Point) -> Point or None:
        """
        Find the closest intersection of a line with an obstacle to a point in 2D space

        Parameters
        ----------
        line: scikit-spatial.objects.Line
            The line with which we are looking for intersection
        point: scikit-spatial.objects.Point
            The 2D point to which we find the nearest intersection

        Returns
        -------
        scikit-spatial.objects.Point
            2D point of the closest intersection. Or None if there are no intersections
        """
        pass


class Circle(Obstacle):
    """
    Static 2D circular obstacle class
    """
    def __init__(self, point: Point, size: float):
        """
        Parameters
        ----------
        point: scikit-spatial.objects.Point
            Circle center point
        size: float
            Size of circle
        """
        self.point = point
        self.size = size
        self.obj = sks_obj.Circle(point, size)

    @staticmethod
    def get_random(area, min_size: float, max_size: float, excluded_area: List):
        wight = area[0]
        height = area[1]
        k = 0
        while True:
            if k > 50:
                raise ValueError("Area generate Error")
            x = np.random.uniform(0, wight)
            y = np.random.uniform(0, height)
            size = np.random.uniform(min_size, max_size)
            flag = False
            for avoid in excluded_area:
                if avoid[0][0] - size < x < avoid[1][0] + size and avoid[0][1] - size < y < avoid[1][1] + size:
                    k += 1
                    flag = True
                    break
            if not flag:
                break
        return Circle(Point([x, y]), size)

    def intersection(self, line: sks_obj.Line, point: Point) -> Point or None:
        try:
            a, b = self.obj.intersect_line(line)
        except ValueError:
            return None
        if np.linalg.norm(a - point) < np.linalg.norm(b - point):
            return a
        else:
            return b
