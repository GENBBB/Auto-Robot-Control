from typing import Self, Optional

import numpy as np
from skspatial.objects import Point

from .controller import Controller
from .lidar import Lidar
from expanse.area import Area
from expanse.obstacles import Circle


class Cluster:
    """
    Robot group class for homogeneous control
    When creating a cluster, the same sampling rate is required for all robots
    And also so that the robots do not collide with each other
    """
    def __init__(self, n: int, size: float, frequency: float,
                 lidar_settings: Optional[dict] = None, controller_settings: Optional[dict] = None) -> None:
        self.x = np.empty((n, 2))
        self.v = np.zeros((n, 2))
        self.turn = np.zeros(n)

        self.size = size
        self.lidar = Lidar(lidar_settings)
        self.controller = Controller(controller_settings)

        self.t = 1 / frequency
        self.frequency = frequency
        self.steps = 0

        self.track = []
        self.angles = []
        self.detected_points = []

    def is_collision(self, area: Optional[Area]) -> bool:
        for i in range(len(self.x) - 1):
            for j in range(i + 1, len(self.x)):
                if np.linalg.norm(self.x[i] - self.x[j]) < 2 * self.size:
                    print("There was a collision with a robot")
                    return True
        if area is not None:
            for i in range(len(self.x)):
                for obj in area.obstacles:
                    if type(obj) is Circle:
                        if np.linalg.norm(self.x[i] - obj.point) < self.size + obj.size:
                            print("There was a collision with a obstacle")
                            return True
        return False

    def control(self, area: Area, target: Point) -> np.ndarray:
        detected_points = self.lidar.scan(area, self.x, self.turn)
        u = self.controller.control(self.x, self.v, detected_points, target)
        self.detected_points.append(detected_points.reshape(-1, 2))
        return u

    def update(self, area: Area, target: Point) -> Self:
        u = self.control(area, target)
        self.x = self.x + self.v * self.t + u * self.t * self.t / 2
        self.v = self.v + u * self.t
        self.turn = np.arctan2(self.v[:, 1], self.v[:, 0])

        self.track.append(self.x)
        self.angles.append(self.turn)
        self.steps += 1
        return self

    def is_coming(self, target: Point, size: float) -> bool:
        diff = target - self.x
        distance = np.linalg.norm(diff, axis=1)
        res = np.all(distance < size)
        if res:
            print("Cluster is coming")
        return np.all(distance < size)

    def parse_trace(self, fps: int):
        frames = [i for i in range(0, self.steps, round(self.frequency / fps))]
        track = [self.track[i] for i in frames]
        angles = [self.angles[i] for i in frames]
        detected_points = [self.detected_points[i] for i in frames]
        return track, angles, detected_points, len(frames)

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