from typing import Self

import math
import numpy as np
from skspatial.objects import Line, Point, Vector

from config import config
from expanse.area import Area
from expanse.obstacles import Circle
import geometry.geometry as geo

max_dv = float(config['Robot']['max_dv'])
gamma_const = [4.0, 4.0]
beta_const = [5.0, 0.0]
alpha_const = [0.5, 0.5]
delta_const = [50.0, 50.0]
rep_const = 1.0
h = 0.5
eps = 0.00001
alpha1 = 1
b = 10
distance = 6
distance_obj = 5
distance_rob = 3
flag_convex = True

dt_vision_segment = np.dtype([('max_angle', float), ('min_angle', float), ('max_point', float, (2, )),
                             ('mid_point', float, (2, )), ('min_point', float, (2, ))])


def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None


class Robot:
    """
    Base class of robot
    """
    def __init__(self, point: Point, radius: float, size: float, lidar_parts: int, frequency: float,
                 v: Vector = None, a: Vector = None):
        """
        Parameters
        ----------
        point : scikit-spatial.objects.Point
            2D point of the robot's initial coordinates
        radius : float
            Robot viewing radius
        size : float
            Robot size
        lidar_parts: int
            The number of rays with which the robot surveys space.
            The angles of the rays are uniformly spaced in the interval from [0, 2pi]
        frequency: int
            The sampling rate at which the robot is updated
            It is believed that with each call a period corresponding to the sampling frequency passes
            The robot also instantly scans the area accessible to it once per sampling period
        v: scikit-spatial.objects.Vector, optional
            Robot initial speed, vector in 2D space
            If None (default), initial speed is zero
        """
        self.position = point
        self.radius = radius
        self.size = size
        self.lidar_parts = lidar_parts
        self.lidar_angle = 2 * math.pi / lidar_parts
        self.t = 1 / frequency
        if v is None:
            v = Vector([0, 0])
        if a is None:
            a = Vector([0, 0])
        self.v = v
        self.a = a
        self.trajectory = []
        self.detected_points = [[]]
        self.angle = [np.arctan2(self.v[1], self.v[0])]
        self.detection_line = [[]]

    def to_convex(self, points, mask_prev):
        mask = mask_prev.astype(int)
        for i in range(self.lidar_parts):
            if mask[i] == 0 and mask[i - 1] == 1:
                last_id = i
                flag = False
                for j in range(5):
                    if mask[(i + j) % self.lidar_parts] == 1:
                        flag = True
                        last_id = (i + j) % self.lidar_parts
                if flag:
                    for j in range(5):
                        if (i + j) % self.lidar_parts != last_id:
                            mask[(i + j) % self.lidar_parts] = 3

        id_start = index(mask, 0)[0]
        id_cur = id_start + 1
        while id_cur != id_start:
            id_cur = id_cur % self.lidar_parts
            if mask[id_cur]:
                id_a = id_cur
                while True:
                    id_b = (id_a + 1) % self.lidar_parts
                    while mask[id_b] == 3:
                        id_b = (id_b + 1) % self.lidar_parts
                    if mask[id_b]:
                        id_c = (id_b + 1) % self.lidar_parts
                        while mask[id_c] == 3:
                            id_c = (id_c + 1) % self.lidar_parts
                        if mask[id_c]:
                            if np.sign(geo.pseudo_dot(points[id_b] - points[id_a], points[id_c] - points[id_b])) != -1:
                                mask[id_b] = 3
                                id_a = id_cur
                            else:
                                id_a = id_b
                        else:
                            flag = False
                            for i in range(1, 5):
                                if mask[(i + id_c) % self.lidar_parts] != 0:
                                    flag = True
                                    break
                            if flag:
                                for i in range(5):
                                    mask[(i + id_c) % self.lidar_parts] = 3
                            else:
                                id_cur = id_c
                                break
                    else:
                        id_cur = id_b
                        break
            else:
                id_cur = (id_cur + 1) % self.lidar_parts
        id_cur = (id_start + 1) % self.lidar_parts
        while id_cur != id_start:
            if mask[id_cur] == 3:
                a_0 = points[id_cur - 1]
                id_tmp = (id_cur + 1) % self.lidar_parts
                while mask[id_tmp] != 1:
                    id_tmp = (id_tmp + 1) % self.lidar_parts
                b_0 = points[id_tmp]
                line1 = Line(a_0, b_0 - a_0)
                while mask[id_cur] == 3:
                    line2 = Line(self.position, [np.cos(id_cur * self.lidar_angle), np.sin(id_cur * self.lidar_angle)])
                    points[id_cur] = line1.intersect_line(line2)
                    id_cur = (id_cur + 1) % self.lidar_parts
            else:
                id_cur = (id_cur + 1) % self.lidar_parts
        return points

    @staticmethod
    def gamma_function(z: float) -> float:
        return z / np.sqrt(1 + z ** 2)

    @staticmethod
    def bump_function(z: float):
        if 0 <= z < h:
            return 1
        elif h <= z <= 1:
            return 0.5 * (1 + np.cos(np.pi * ((z - h) / (1 - h))))
        else:
            return 0

    @staticmethod
    def sigma_norm(z: float) -> float:
        return (np.sqrt(1 + eps * z ** 2) - 1) / eps

    def repulsive_function(self, z: float, d) -> float:
        return self.bump_function(z / self.sigma_norm(d)) * (self.gamma_function(z - self.sigma_norm(d)) - 1)

    def alpha_function(self, z: float) -> float:
        return 0.5 * ((alpha1 + b) * self.gamma_function(z + (b - alpha1) / np.sqrt(4 * alpha1 * b)) + alpha1 - b)

    def attractive_functions(self, z: float) -> float:
        return self.bump_function(z / self.sigma_norm(self.radius)) * self.alpha_function(z - self.sigma_norm(distance))

    def alpha_control(self, cluster):
        sum1 = np.zeros(2)
        sum2 = np.zeros(2)
        for robot in cluster:
            if robot is not self:
                diff = robot.position - self.position
                norm = np.linalg.norm(diff)
                sum1 += self.attractive_functions(self.sigma_norm(norm)) * diff / np.sqrt(1 + eps * norm ** 2)
                sum2 += self.bump_function(self.sigma_norm(norm) / self.sigma_norm(self.radius)) * (robot.v - self.v)
        return alpha_const[0] * sum1 + alpha_const[1] * sum2

    def beta_control(self, detected_points: list[Point]) -> Vector:
        sum1 = np.zeros(2)
        sum2 = np.zeros(2)
        for point in detected_points:
            mu = 0.1 / np.linalg.norm(self.position - point)
            q = mu * self.position + (1 - mu) * point
            diff = q - self.position
            a = np.reshape((self.position - point) / np.linalg.norm(self.position - point), (2, 1))
            p = mu * np.matmul((np.eye(2) - np.matmul(a, np.transpose(a))), self.v)
            norm = np.linalg.norm(diff)
            sum1 += self.repulsive_function(self.sigma_norm(norm), distance_obj) * diff / np.sqrt(1 + eps * norm ** 2)
            sum2 += self.bump_function(self.sigma_norm(norm) / self.sigma_norm(distance_obj)) * (p - self.v)
        return beta_const[0] * sum1 + beta_const[1] * sum2

    def gamma_control(self, end_area: Point) -> Vector:
        diff = self.position - end_area
        return -gamma_const[0] * diff / np.sqrt(1 + np.linalg.norm(diff)) - gamma_const[1] * self.v

    def delta_control(self, cluster):
        sum1 = np.zeros(2)
        sum2 = np.zeros(2)
        for robot in cluster:
            diff = robot.position - self.position
            norm = np.linalg.norm(diff)
            sum1 += self.repulsive_function(self.sigma_norm(norm), distance_rob) * diff / np.sqrt(1 + eps * norm ** 2)
            sum2 += self.bump_function(self.sigma_norm(norm) / self.sigma_norm(distance_rob)) * (robot.v - self.v)
        return delta_const[0] * sum1 + delta_const[1] * sum2

    def control(self, area: Area, end_area: Point, cluster: list[Self]) -> Vector:
        """
        Control the speed of the robot, taking into account the area in which it moves

        Parameters
        -------
        area: expanse.Area
            An area with obstacles through which the robot moves

        Returns
        -------
        Vector
            Updated robot speed
        """
        """In Progress"""
        detected_points = self.lidar(area)
        return self.alpha_control(cluster) + self.beta_control(detected_points) + self.gamma_control(end_area) + self.delta_control(cluster)

    def update(self, area: Area, end_area: Point, cluster: list[Self]) -> Self:
        """
        Update robot speed and coordinates according to sampling rate. Also saves the trajectory for robot
        Returns
        -------
        self: Robot
            Robot in updated condition
        """
        self.trajectory.append(self.position)
        self.a = self.control(area, end_area, cluster)
        self.position = self.position + self.t * self.v + self.t * self.t * self.a / 2
        self.v = self.v + self.a * self.t
        self.angle.append(np.arctan2(self.v[1], self.v[0]))
        return self