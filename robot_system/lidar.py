import math
from typing import Any

from numpy import ndarray, dtype
from skspatial.objects import Point, Line
import numpy as np

from expanse import Circle, Area
import geometry as geo


class Lidar:
    def __init__(self, lidar_settings: dict) -> None:
        self.radius = lidar_settings['radius']
        self.lidar_parts = lidar_settings['lidar_parts']
        self.lidar_angle = 2 * math.pi / self.lidar_parts
        self.rays = np.full((self.lidar_parts, 2), Point([0, 0]))

    def detected_circle(self, circle: Circle, pos, turn, rays: np.ndarray) -> np.ndarray or None:
        if circle.point.distance_point(pos) <= self.radius + circle.size:
            try:
                min_angle, max_angle = geo.viewing_angels_circle(pos, circle.point, circle.size)
            except ValueError:
                return None
            alpha = math.floor((min_angle - turn) / self.lidar_angle)
            beta = math.ceil((max_angle - turn) / self.lidar_angle)
            angles = np.linspace(alpha * self.lidar_angle + turn, beta * self.lidar_angle + turn, beta - alpha + 1)
            for i in range(0, beta - alpha + 1):
                line = Line(pos, [np.cos(angles[i]), np.sin(angles[i])])
                intersect = circle.intersection(line, pos)
                ind = (i+alpha) % self.lidar_parts
                if intersect is not None and intersect.distance_point(pos) <= np.linalg.norm(pos - rays[ind]):
                    rays[ind] = intersect
        return rays

    def scan(self, area: Area, x: np.ndarray, turn: np.ndarray, size: float) -> np.ndarray:
        self.rays = np.empty((len(x), self.lidar_parts, 2))
        for i in range(len(self.rays)):
            self.rays[i] = np.full((self.lidar_parts, 2), x[i] + [2 * self.radius, 0])
        for i in range(len(self.rays)):
            for obj in area.obstacles:
                if type(obj) is Circle:
                    self.rays[i] = self.detected_circle(obj, x[i], turn[i], self.rays[i])
        return self.rays