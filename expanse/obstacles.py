from config import config
from abc import ABC, abstractmethod
from typing import Self

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
    def get_random(min_size: float, max_size: float, range_x: float or (float, float),
                   range_y: float or (float, float) = None):
        """
        Static method returning a random object of the circle class
        A circle is created with the center coordinate in the interval [min_x, max_x] * [min_y, max_y]
        and size in the interval [min_size, max_size]

        Parameters
        -------
        min_size: float
            Lower limit of circle size.
        max_size: float
            Upper limit of circle size.
        range_x: float or tuple of float
            Range of boundaries of the circle center in X. If float is specified, then the range is [0, range_x].
            If a tuple of the form (min_x, max_x) is specified, then the range is [min_x, max_x]
        range_y: float or tuple of float or None
            Range of boundaries of the circle center in Y. If float is specified, then the range is [0, range_y].
            If a tuple of the form (min_y, max_y) is specified, then the range is [min_y, max_y]
            If range_y is None(default), the range in Y is the same as in X

        Returns
        -------
        Circle
            Random circle with given parameters
        """
        if range_y is None or range_x == range_y:
            if type(range_x) is tuple:
                min_x = range_x[0]
                max_x = range_x[1]
            else:
                min_x = 0
                max_x = range_x
            return Circle(Point(np.random.uniform(min_x, max_x, 2)), np.random.uniform(min_size, max_size))
        else:
            if type(range_x) is tuple:
                min_x = range_x[0]
                max_x = range_x[1]
            else:
                min_x = 0
                max_x = range_x
            if type(range_y) is tuple:
                min_y = range_y[0]
                max_y = range_y[1]
            else:
                min_y = 0
                max_y = range_y
            return Circle(Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]),
                          np.random.uniform(min_size, max_size))

    def intersection(self, line: sks_obj.Line, point: Point) -> Point or None:
        try:
            a, b = self.obj.intersect_line(line)
        except ValueError:
            return None
        if np.linalg.norm(a - point) < np.linalg.norm(b - point):
            return a
        else:
            return b
