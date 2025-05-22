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

class DinCircles:
    def __init__(self, x, size, frequency):
        self.x = x
        self.size = size
        self.v = np.zeros(self.x.shape)
        self.T = 1 / frequency
        self.frequency = frequency
        self.static = False

        self.track = []
        self.steps = 0

    def update(self):
        self.steps += 1
        self.track.append(self.x)
        if self.static:
            return
        u = np.random.normal(0, 10, self.x.shape)
        self.x = self.x + self.T * self.v + self.T * self.T * u / 2
        self.v = self.v + self.T * u
        mask_x = (self.x[:, 0] < 0) | (self.x[:, 0] > 100)
        self.v[mask_x, 0] *= -1
        mask_y = (self.x[:, 1] < 0) | (self.x[:, 1] > 100)
        self.v[mask_y, 1] *= -1
        self.v *= 0.99

    @staticmethod
    def get_random(area, min_size: float, max_size: float, excluded_area: List, n: int, frequency: float):
        cx = np.empty((n, 2))
        size = np.empty(n)
        wight = area[0]
        height = area[1]
        for i in range(n):
            k = 0
            while True:
                if k > 50:
                    raise ValueError("Area generate Error")
                x = np.random.uniform(0, wight)
                y = np.random.uniform(0, height)
                r = np.random.uniform(min_size, max_size)
                flag = False
                for avoid in excluded_area:
                    if avoid[0][0] - r < x < avoid[1][0] + r and avoid[0][1] - r < y < avoid[1][1] + r:
                        k += 1
                        flag = True
                        break
                if not flag:
                    break
            cx[i] = [x, y]
            size[i] = r
        return DinCircles(cx, size, frequency)

    def parse(self, fps):
        if self.steps == 0:
            return [[[0, 0]]], [[0]], 0
        frames = [i for i in range(0, self.steps, round(self.frequency / fps))]
        track = [self.track[i] for i in frames]
        return track, self.size, len(frames)

class LetterC:
    def __init__(self, size, frequency, c, turn, center_z):
        self.size = size
        self.v = np.zeros(c.shape)
        self.T = 1 / frequency
        self.frequency = frequency
        self.n = len(c) // 5
        self.track = []
        self.steps = 0
        self.c = c
        self.turn = turn
        self.omega = np.zeros(turn.shape)
        self.center_z = center_z
        self.x = self.get_x(c, turn, center_z)

    @staticmethod
    def get_x(c, turn, center_z):
        cos_theta = np.cos(turn)
        sin_theta = np.sin(turn)
        x = center_z[:, 0]
        y = center_z[:, 1]

        new_x = x * cos_theta - y * sin_theta
        new_y = x * sin_theta + y * cos_theta

        return np.column_stack((new_x, new_y)) + c


    def update(self):
        self.steps += 1
        self.track.append(self.x)
        u = np.random.normal(0, 10, (self.n, 2))
        u = np.repeat(u, 5, axis=0)
        self.c = self.c + self.T * self.v + self.T * self.T * u / 2
        self.v = self.v + self.T * u
        mask_x = (self.c[::5, 0] < 0) | (self.c[::5, 0] > 100)
        mask_x = np.repeat(mask_x, 5, axis=0)
        self.v[mask_x, 0] *= -1
        mask_y = (self.c[::5, 1] < 0) | (self.c[::5, 1] > 100)
        mask_y = np.repeat(mask_y, 5, axis=0)
        self.v[mask_y, 1] *= -1
        self.v *= 0.99

        theta = np.random.normal(0, np.pi, self.n)
        theta = np.repeat(theta, 5, axis=0)
        self.turn = self.turn + self.T * self.omega + theta * self.T * self.T / 2
        self.omega = self.omega + self.T * theta
        self.x = self.get_x(self.c, self.turn, self.center_z)
        self.omega *= 0.98


    @staticmethod
    def _calculate_radius(R_center):
        # Рассчитываем минимальный радиус для пересечения соседних кругов
        angle_between = 25  # Угол между соседними центрами
        theta = np.radians(angle_between / 2)
        min_radius = R_center * np.sin(theta)
        return min_radius * 1.65  # Добавляем 5% для гарантированного пересечения

    @staticmethod
    def _calculate_centers(R_center):
        # Располагаем центры по дуге 180°
        angles = np.linspace(90, 270, 5)
        res = [(R_center * np.cos(np.radians(a)),
                 R_center * np.sin(np.radians(a))) for a in angles]
        res = np.array(res)
        return res

    @staticmethod
    def get_random(area, min_size: float, max_size: float, excluded_area: List, n: int, frequency: float):
        cx = np.empty((5 * n, 2))
        r = np.empty(5 * n)
        wight = area[0]
        height = area[1]
        c = np.empty((5 * n, 2))
        turn = np.empty(5 * n)
        c_z = np.empty((5 * n, 2))
        for i in range(n):
            k = 0
            while True:
                if k > 50:
                    raise ValueError("Area generate Error")
                x = np.random.uniform(0, wight)
                y = np.random.uniform(0, height)
                R_center = np.random.uniform(min_size, max_size)
                circle_radius = LetterC._calculate_radius(R_center)
                centers = LetterC._calculate_centers(R_center)
                theta = np.random.uniform(0, 2 * np.pi)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                rotation_matrix = np.array([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])
                centers_z = np.dot(centers, rotation_matrix.T)
                centers = centers_z + [x, y]
                size = circle_radius
                flag = False
                for avoid in excluded_area:
                    for x, y in centers:
                        if avoid[0][0] - size < x < avoid[1][0] + size and avoid[0][1] - size < y < avoid[1][1] + size:
                            k += 1
                            flag = True
                            break
                    if flag:
                        break
                if not flag:
                    break
            turn[5*i:5*i+5] = theta
            c[5*i:5*i+5] = [x, y]
            cx[5*i:5*i+5] = centers
            r[5*i:5*i+5] = circle_radius
            c_z[5*i:5*i+5] = centers_z
        return LetterC(r, frequency, c, turn, c_z)

    def parse(self, fps):
        if self.steps == 0:
            return [[[0, 0]]], [[0]], 0
        frames = [i for i in range(0, self.steps, round(self.frequency / fps))]
        track = [self.track[i] for i in frames]
        return track, self.size, len(frames)