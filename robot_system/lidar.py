import numpy as np
import cupy as cp

from expanse import Circle, Area


class Lidar:
    def __init__(self, lidar_settings: dict) -> None:
        self.xr = np

        self.radius = lidar_settings['radius']
        self.n_rays = lidar_settings['lidar_parts']
        self.lidar_angle = 2 * np.pi / self.n_rays

        self.angles = self.xr.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        self.directions = None
        self.intersections = None

    def scan_circles(self, c_x: np.ndarray, c_r: np.ndarray, x: np.ndarray) -> np.ndarray:
        if c_x.shape[0] == 0:
            intersections = x[:, None, :] + 2 * self.radius * self.directions
            return intersections
        oc = c_x[:, None, None, :] - x[None, :, None, :]
        directions = self.xr.tile(self.directions, (c_x.shape[0], 1, 1, 1))
        proj = self.xr.einsum('ijkl,ijkl->ijk', oc, directions)

        oc_norm2 = self.xr.sum(oc ** 2, axis=3)
        #print(c_r)
        r2 = c_r[:, None, None] ** 2
        disc = proj ** 2 - oc_norm2 + r2

        valid = disc >= 0
        sqrt_disc = self.xr.sqrt(self.xr.maximum(disc, 0))

        t1 = proj - sqrt_disc
        t2 = proj + sqrt_disc

        t_min = self.xr.where((0 < t1) & (t1 < self.radius) & valid, t1,
                         self.xr.where((0 < t2) & (t2 < self.radius) & valid, t2, 2 * self.radius))

        best_t = self.xr.min(t_min, axis=0)
        intersections = x[:, None, :] + best_t[..., None] * self.directions

        return intersections

    def scan(self, area: Area, x: np.ndarray, turn: np.ndarray) -> np.ndarray:
      #  self.intersections = np.full((x.shape[0], self.n_rays, 2), np.inf)
        turn = turn.reshape(-1, 1)
        rotated_angles = self.angles + turn
        self.directions = self.xr.stack((np.cos(rotated_angles), np.sin(rotated_angles)), axis=-1)
        self.intersections = self.directions * self.radius + x.reshape(x.shape[0], -1, x.shape[1])

        c_x = area.obstacles.x
        c_r = area.obstacles.size
        c_x = np.array(c_x)
        c_r = np.array(c_r)
        intersections = self.scan_circles(c_x, c_r, x)
        return intersections