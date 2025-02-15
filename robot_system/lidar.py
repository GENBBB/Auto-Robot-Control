import math
from typing import Any

from numpy import ndarray, dtype
from scipy.optimize import direct
from skspatial.objects import Point, Line
import numpy as np

from expanse import Circle, Area
import geometry as geo


class Lidar:
    def __init__(self, lidar_settings: dict) -> None:
        self.radius = lidar_settings['radius']
        self.n_rays = lidar_settings['lidar_parts']
        self.lidar_angle = 2 * math.pi / self.n_rays

        self.angles = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        self.directions = None
        self.intersections = None

    def find_intersections(self, c_x: np.ndarray, c_r: np.ndarray, x: np.ndarray) -> np.ndarray:
        oc = c_x[:, None, None, :] - x[None, :, None, :]
        directions = np.tile(self.directions, (c_x.shape[0], 1, 1, 1))
        proj = np.einsum('ijkl,ijkl->ijk', oc, directions)

        oc_norm2 = np.sum(oc ** 2, axis=3)
        r2 = c_r[:, None, None] ** 2
        disc = proj ** 2 - oc_norm2 + r2

        valid = disc >= 0
        sqrt_disc = np.sqrt(np.maximum(disc, 0))

        t1 = proj - sqrt_disc
        t2 = proj + sqrt_disc

        t_min = np.where((0 < t1) & (t1 < self.radius) & valid, t1,
                         np.where((0 < t2) & (t2 < self.radius) & valid, t2, np.inf))

        best_t = np.min(t_min, axis=0)
        intersections = x[:, None, :] + best_t[..., None] * directions

        return intersections

    def scan(self, area: Area, x: np.ndarray, turn: np.ndarray) -> np.ndarray:
        self.intersections = np.full((x.shape[0], self.n_rays, 2), np.inf)
        turn = turn.reshape(-1, 1)
        rotated_angles = self.angles + turn
        self.directions = np.stack((np.cos(rotated_angles), np.sin(rotated_angles)), axis=-1)

        c_x = []
        c_r = []
        for obj in area.obstacles:
            if type(obj) is Circle:
                c_x.append(obj.point)
                c_r.append(obj.size)
        c_x = np.array(c_x)
        c_r = np.array(c_r)
        return self.find_intersections(c_x, c_r, x)