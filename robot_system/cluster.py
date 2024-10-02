from typing import Self, Optional, List, NoReturn

import numpy as np
from skspatial.objects import Point

from robot import Robot
from expanse.area import Area
from expanse.obstacles import Circle
from .trace import Trace


class Cluster:
    """
    Robot group class for homogeneous control
    When creating a cluster, the same sampling rate is required for all robots
    And also so that the robots do not collide with each other
    """
    def __init__(self, robots: List[Robot], frequency: float) -> None:
        """
        Parameters
        ----------
        robots: list of Robot
            List of robots in the cluster
        frequency: float
            Cluster sampling rate
        """
        self.robots = robots
        self.check_collision(None)
        self.frequency = frequency
        self.steps = 0

    def check_collision(self, area: Optional[Area]) -> NoReturn:
        for id1, rob1 in enumerate(self.robots):
            for rob2 in self.robots[id1+1:]:
                if np.linalg.norm(rob1.position - rob2.position) < rob1.size + rob2.size:
                    raise RuntimeError("There was a collision with a robot")
        for rob in self.robots:
            for obj in area.obstacles:
                if type(obj) is Circle:
                    if np.linalg.norm(rob.position - obj.position) < rob.size + obj.size:
                        raise RuntimeError("There was a collision with a obstacle")

    def update(self, area: Area, target: Point) -> Self:
        """
        Moves the cluster during sampling time across the area to the selected target.
        Updates the state of the robots in the cluster according to the sampling rate of each robot.

        Parameters
        -------
        area: Area
            An area with obstacles through which the cluster moves
        target: Point
            The goal of the movement at the current moment in time

        Returns
        -------
        self: Cluster
            Cluster in updated condition
        """
        for robot in self.robots:
            robot.update(area, target, self.robots)
        self.check_collision(area)
        self.steps += 1
        return self

    def parse_trace(self, fps: int) -> List[Trace]:
        """
        Get cluster track for all time according to frames per second
        Returns the Tuple of instances of the Trace class

        Returns
        -------
        list of Trace
            Cluster trace
        """
        frames = [i for i in range(0, self.steps, round(self.frequency / fps))]
        cluster_trace = []
        for robot in self.robots:
            track = []
            angle = []
            detected_points = []
            for frame in frames:
                track.append(robot.trajectory[frame])
                angle.append(robot.angle[frame])
                detected_points.append(robot.detected_points[frame])
            cluster_trace.append(Trace(robot.size, track, angle, detected_points))
        return cluster_trace


def create_cluster(n_robots: int, start_area: Point, size_start_area: float, size: float, radius: float,
                   lidar_parts: int, frequency: float) -> Cluster:
    """
    Create a group of homogeneous robots randomly distributed in a given circle area with zero initial speed

    Parameters
    ----------
    n_robots: int
        Number of robots in the cluster
    start_area: Point
        Coordinates of the center of the circular starting area
    size_start_area: float
        Size of circular starting area
    radius: float
        Radius of robots
    size: float
        Size of robots
    lidar_parts: int
        The number of rays with which the robots surveys space
    frequency: float
        Sampling frequency

    Returns
    -------
    Cluster
        Robot cluster in a given area
    """
    cluster = []
    for i in range(n_robots):
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
            for rob in cluster:
                if np.linalg.norm(rob.position - position) < 2 * size:
                    flag = True
                    break
        cluster.append(Robot(position, radius, size, lidar_parts, frequency))
    return Cluster(cluster, frequency)
