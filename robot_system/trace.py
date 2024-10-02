from typing import List

from skspatial.objects import Point


class Trace:
    """
    Class for storing the Robot trace
    Stores the size of the robot, its trajectory, rotation angles, and detected obstacles
    """
    def __init__(self, size: float, track: List[Point] = None, angle: List[Point] = None,
                 detected_points: List[List[Point]] = None):
        """
        Parameters
        ----------
        size: float
            The size of robot
        track: list of Point, optional
            Robot trajectory at certain points in time according to the sampling time
        angle: list of Point, optional
            Robot trajectory at certain points in time according to the sampling time
        detected_points: list of Point, optional
            List of obstacles detected by the robot at each moment in time according to the sampling time
        """
        self.size = size
        if track is None:
            self.track = track
        if angle is None:
            self.angle = angle
        if detected_points is None:
            self.detected_points = detected_points
