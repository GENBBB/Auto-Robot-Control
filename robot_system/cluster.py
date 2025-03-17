from typing import Self, Optional, List, Tuple
from itertools import chain

import numpy as np

from .splitter import Splitter
from .lidar import Lidar
from .controller import Controller
from expanse.area import Area
from expanse.obstacles import Circle


class Cluster:
    """
    Robot group class for homogeneous control
    When creating a cluster, the same sampling rate is required for all robots
    And also so that the robots do not collide with each other
    """
    def __init__(self, n: int, size: float, frequency: float,
                 lidar_settings: Optional[dict] = None, controller_settings: Optional[dict] = None,
                 splitter_settings: Optional[dict] = None) -> None:
        self.x = np.empty((n, 2))
        self.v = np.zeros((n, 2))
        self.turn = np.zeros(n)

        self.size = size
        self.lidar = Lidar(lidar_settings)
        if controller_settings is not None:
            self.controller = Controller(controller_settings)
        self.splitter = Splitter(splitter_settings)

        self.t = 1 / frequency
        self.frequency = frequency
        self.steps = 0

        self.track = []
        self.angles = []
        self.detected_points = []
        self.beta_track = []

        self.live = np.full(n, True)

        self.target_distance = []
        self.formation_distance = controller_settings['formation_distance']
        self.radius = lidar_settings['radius']
        self.convexity = []

    def is_collision(self, area: Optional[Area]) -> None:
        for i in range(len(self.x) - 1):
            for j in range(i + 1, len(self.x)):
                if np.linalg.norm(self.x[i] - self.x[j]) < 2 * self.size:
                    if self.live[i] or self.live[j]:
                        #print("There was a collision with a robot")
                        self.v[i] = 0
                        self.v[j] = 0
                        self.live[i] = False
                        self.live[j] = False
        if area is not None:
            for i in range(len(self.x)):
                for obj in area.obstacles:
                    if type(obj) is Circle:
                        if np.linalg.norm(self.x[i] - obj.point) < self.size + obj.size:
                            if self.live[i]:
                                #print("There was a collision with a obstacle")
                                self.live[i] = False
                                self.v[i] = 0

    def beta_position(self, obstacles: List[List[np.ndarray]], x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(obstacles) != x.shape[0]:
            raise ValueError("Количество групп оболочек и целевых точек должно совпадать")

        closest_result = []
        direct_result = []
        for group_idx, hull_group in enumerate(obstacles):
            target_point = x[group_idx]
            closest_group = []
            direct_group = []

            for hull in hull_group:
                closest, direct = self.find_closest_on_hull(target_point, hull)
                closest_group.append(closest.reshape(2))
                direct_group.append(direct.reshape(2))

            closest_group = np.array(closest_group).reshape(-1, 2)
            direct_group = np.array(direct_group).reshape(-1, 2)
            closest_result.append(closest_group)
            direct_result.append(direct_group)

        return closest_result, direct_result

    @staticmethod
    def find_closest_on_hull(target: np.ndarray, hull: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = hull
        b = np.roll(hull, -1, axis=0)

        ap = target - a
        ab = b - a
        t = np.sum(ap * ab, axis=1) / (np.sum(ab * ab, axis=1) + 1e-8)
        t_clipped = np.clip(t, 0, 1)
        closest_points = a + t_clipped[:, np.newaxis] * ab

        dists = np.linalg.norm(closest_points - target, axis=1)
        min_idx = np.argmin(dists)
        return closest_points[min_idx], ab[min_idx]

    def control(self, area: Area, target: np.ndarray) -> np.ndarray:
        detected_points = self.lidar.scan(area, self.x[self.live], self.turn[self.live])
        obstacles = self.splitter.obstacle_split(detected_points, self.x[self.live])
        beta_position, beta_direction = self.beta_position(obstacles, self.x[self.live])

        u = self.controller.control(self.x[self.live], self.v[self.live], beta_position, beta_direction, target)

        flat_arrays = list(chain.from_iterable(obstacles))
        if not flat_arrays:
            self.detected_points.append(np.empty((0, 2)))
        else:
            self.detected_points.append(np.concatenate(flat_arrays, axis=0))

        self.beta_track.append(np.vstack(beta_position))
        return u

    def update(self, area: Area, target: np.ndarray) -> Self:
        self.target_distance.append(np.linalg.norm(target - self.x))
        self.convexity.append(self.computate_convexity())
        if not np.any(self.live):
            self.track.append(self.x)
            self.angles.append(self.turn)
            self.detected_points.append(np.empty((0, 2)))
            self.beta_track.append(np.empty((0, 2)))
            self.steps += 1
            return self
        u = self.control(area, target)
        self.x[self.live] = self.x[self.live] + self.v[self.live] * self.t + u * self.t * self.t / 2
        self.v[self.live] = self.v[self.live] + u * self.t
        self.turn[self.live] = np.arctan2(self.v[self.live][:, 1], self.v[self.live][:, 0])
        self.is_collision(area)
        self.track.append(np.copy(self.x))
        self.angles.append(np.copy(self.turn))
        self.steps += 1
        return self

    def is_coming(self, target: np.ndarray, size: float) -> bool:
        if not np.any(self.live):
            return False
        diff = target - self.x[self.live]
        distance = np.linalg.norm(diff, axis=1)
        res = np.all(distance < size) & np.all(np.linalg.norm(self.v[self.live]) < 0.5)
        if res:
            print("Cluster is coming")
        return res

    def parse_trace(self, fps: int):
        if self.steps == 0:
            return [[[0, 0]]], [[0]], [np.empty((0, 2))], [np.empty((0, 2))], 0
        frames = [i for i in range(0, self.steps, round(self.frequency / fps))]
        track = [self.track[i] for i in frames]
        angles = [self.angles[i] for i in frames]
        detected_points = [self.detected_points[i] for i in frames]
        beta_track = [self.beta_track[i] for i in frames]
        return track, angles, detected_points, beta_track, len(frames)

    @staticmethod
    def st_arrangement(n_robots: int, size_robots: int, start_area: np.ndarray,
                       size_start_area: float) -> Tuple[np.ndarray, np.ndarray]:
        x = np.empty((n_robots, 2))
        turn = np.empty(n_robots)
        for i in range(n_robots):
            flag = True
            k = 0
            position = None
            while flag:
                if k > 100:
                    raise RuntimeWarning
                r = size_start_area * np.sqrt(np.random.uniform())
                theta = np.random.uniform() * 2 * np.pi
                position = start_area + r * np.array([np.cos(theta), np.sin(theta)])
                flag = False
                k += 1
                for j in range(i):
                    if np.linalg.norm(x[j] - position) < 2 * size_robots:
                        flag = True
                        break
            x[i] = position
            turn[i] = np.random.uniform() * 2 * np.pi
        return x, turn

    def arrangement(self, x: np.ndarray, turn: float) -> Self:
        self.x = x
        self.turn = turn
        self.track.append(self.x)
        self.angles.append(self.turn)
        self.detected_points.append(np.empty((0, 2)))
        return self

    def get_time(self):
        return self.steps

    def get_target_distance(self):
        return np.mean(self.target_distance)

    def get_dead(self):
        return 1 - np.mean(self.live)

    def computate_convexity(self):
        if not np.any(self.live):
            return 0
        x = self.x[self.live]
        diff_q = np.full((len(x), len(x), 2), x)
        diff_q = diff_q - np.transpose(diff_q, axes=(1, 0, 2))
        diff_q = np.linalg.norm(diff_q, axis=2)
        diff = np.abs(diff_q - self.formation_distance)
        diff[diff_q == 0] = 0
        diff[diff_q > self.radius] = 0
        return np.mean(diff)

    def get_convexity(self):
        return np.mean(self.convexity)