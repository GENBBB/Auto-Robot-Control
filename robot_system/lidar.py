import math
from skspatial.objects import Point, Line
import numpy as np

from expanse import Circle
import geometry as geo


class Lidar:
    """

    """
    def __init__(self, radius: float, lidar_parts: int) -> None:
        """
        Parameters
        ----------
        radius : float
            Robot viewing radius
        lidar_parts: int
            The number of rays with which the robot surveys space.
            The angles of the rays are uniformly spaced in the interval from [0, 2pi]
        """
        self.radius = radius
        self.lidar_parts = lidar_parts
        self.lidar_angle = 2 * math.pi / lidar_parts
        self.rays = np.full(lidar_parts, Point([0, 0]))

    def detected_circle(self, circle: Circle, pos: Point, turn: float) -> list[Point]:
        """
        Detection of circle points visible to the robot

        Parameters
        ----------
        circle: expanse.Circle
            A circle whose points must be discovered
        pos: Point
            Current robot position
        turn: float
            Robot rotation angle

        Returns
        -------
        list of Point
            List of detected circle points
        """
        if circle.point.distance_point(pos) <= self.radius + circle.size:
            try:
                min_angle, max_angle = geo.viewing_angels_circle(pos, circle.point, circle.size)
            except ValueError:
                return []
            alpha = math.floor((min_angle - turn) / self.lidar_angle)
            beta = math.ceil((max_angle - turn) / self.lidar_angle)
            angles = np.linspace(alpha * self.lidar_angle + turn, beta * self.lidar_angle + turn, beta - alpha + 1)
            for i in range(0, beta - alpha + 1):
                line = Line(pos, [np.cos(angles[i]), np.sin(angles[i])])
                intersect = circle.intersection(line, pos)
                if intersect is not None and intersect.distance_point(pos) <= self.rays[i + alpha].distance_point(pos):
                    self.rays[i + alpha] = intersect
        return self.rays

    def lidar(self, area: Area, pos: Point, turn: float, size: float) -> list[Point]:
        """
        Detection of obstacles visible to the robot.
        Also saves a list of visible points in the attribute detected_points
        That is a list of visible points at each step

        Parameters
        ----------
        area: Area
            An area with obstacles through which the robot moves
        pos: Point
            Current robot position
        turn: float
            Robot rotation angle

        Returns
        -------
        list of Point
            List of points of obstacles visible to the robot
        """
        self.rays = np.full(self.lidar_parts, pos + [self.radius, 0])
        for obj in area.obstacles:
            if type(obj) is Circle:
                self.detected_circle(obj, pos, turn)
        tmp = np.full(self.rays.shape, pos)
        mask1 = np.linalg.norm(self.rays - tmp, axis=1) <= self.radius
        mask2 = np.linalg.norm(self.rays - tmp, axis=1) >= size
        if flag_convex:
            self.rays = self.to_convex(self.detected_points[-1], mask1 & mask2)
            if np.linalg.norm(self.a) < 0.1:
                self.detected_points[-1] = self.to_convex(self.detected_points[-1], mask1 & mask2)
        self.detected_points[-1] = self.detected_points[-1][mask1 & mask2]
        return self.detected_points[-1]
