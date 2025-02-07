import math
from typing import Self, Optional, List, NoReturn

import numpy as np
from skspatial.objects import Point

from .lidar import Lidar
from .robot import Robot
from expanse.area import Area
from expanse.obstacles import Circle
from .trace import Trace


class Cluster:
    """
    Robot group class for homogeneous control
    When creating a cluster, the same sampling rate is required for all robots
    And also so that the robots do not collide with each other
    """
    def __init__(self, n: int, size: float, radius: float, lidar_parts: int, frequency: float) -> None:
        self.x = np.empty((n, 2))
        self.v = np.zeros((n, 2))
        self.turn = np.empty(n)

        self.size = size
        self.lidar = Lidar(radius, lidar_parts)

        self.t = 1 / frequency
        self.frequency = frequency
        self.steps = 0

        self.track = []
        self.angles = []
        self.detected_points = []

    def check_collision(self, area: Optional[Area]) -> NoReturn:
        for i in range(len(self.x) - 1):
            for j in range(i + 1, len(self.x)):
                if np.linalg.norm(self.x[i] - self.x[j]) < 2 * self.size:
                    raise RuntimeError("There was a collision with a robot")
        if area is not None:
            for i in range(len(self.x)):
                for obj in area.obstacles:
                    if type(obj) is Circle:
                        if np.linalg.norm(self.x[i] - obj.point) < self.size + obj.size:
                            raise RuntimeError("There was a collision with a obstacle")

    def control(self, area: Area) -> np.ndarray:
        detected_points = self.lidar.scan(area, self.x, self.turn, self.size)
        self.detected_points.append(detected_points.reshape(-1, 2))
        u = np.ones((len(self.x), 2))
        return u

    def update(self, area: Area, target: Point) -> Self:
        u = self.control(area)
        self.x = self.x + self.v * self.t + u * self.t * self.t / 2
        self.v = self.v + u * self.t
        self.turn = np.arctan2(self.v[:, 1], self.v[:, 0])

        self.track.append(self.x)
        self.angles.append(self.turn)
        self.check_collision(area)
        # noinspection PyUnreachableCode
        self.steps += 1
        return self

    def parse_trace(self, fps: int):
        frames = [i for i in range(0, self.steps, round(self.frequency / fps))]
        track = [self.track[i] for i in frames]
        angles = [self.angles[i] for i in frames]
        detected_points = [self.detected_points[i] for i in frames]
        return track, angles, detected_points

    def arrangement(self, start_area: Point, size_start_area: float) -> Self:
        for i in range(len(self.x)):
            flag = True
            k = 0
            position = None
            while flag:
                if k > 100:
                    raise RuntimeWarning
                r = size_start_area * np.sqrt(np.random.uniform())
                theta = np.random.uniform() * 2 * np.pi
                position = Point(start_area + r * np.array([np.cos(theta), np.sin(theta)]))
                flag = False
                k += 1
                for j in range(i):
                    if np.linalg.norm(self.x[j] - position) < 2 * self.size:
                        flag = True
                        break
            self.x[i] = position
            self.turn[i] = np.random.uniform() * 2 * np.pi
        self.track.append(self.x)
        self.angles.append(self.turn)
        self.detected_points.append(np.empty((0, 2)))
        return self