import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.collections as collections
from matplotlib.patches import RegularPolygon
from config import config
from expanse.area import Area
from robot_system import Cluster
import numpy as np
from typing import Self
import math

interval = 30
width = float(config['Config']['width'])
height = float(config['Config']['height'])


class SystemAnimation:
    """
    Class for animating the movement of a cluster in an area with obstacles
    """
    def __init__(self, area: Area, cluster: Cluster, steps: int, collision_frame: int, robot_vision: bool = False):
        """
        Parameters
        ----------
        area: expanse.Area
            The area over which the cluster moved after the end of the movement
        cluster: robot_system.Cluster
            Cluster of robots after the end of the movement
        steps: int
            Number of steps taken by a cluster and area
        """
        self.static_circle = area.parse()
        self.cluster_trace, self.detected_points_trace, self.detection_line_trace = cluster.parse_trace(30)
        self.cluster_collections = None
        self.cluster = None
        self.fig = None
        self.ax = None
        self.scatter = None
        self.line_collection = None
        self.line = None
        self.robots_vision = robot_vision
        if not self.robots_vision:
            self.detected_points_trace = None
        self.collision_frame = collision_frame
        self.frames = steps

    def start(self) -> Self:
        """
        Running cluster motion animation
        Returns
        -------

        """
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlim=[0, width], ylim=[0, height], xticks=range(0, int(width), 5), yticks=range(0, int(height), 5))
        # Drawing map
        circle_coll = collections.EllipseCollection(2 * self.static_circle['size'], 2 * self.static_circle['size'],
                                                    angles=0, units='xy',
                                                    offsets=self.static_circle['coordinates'],
                                                    offset_transform=self.ax.transData)
        self.ax.add_collection(circle_coll)

        # Drawing robots
        self.cluster_collections = []
        self.frames = len(self.cluster_trace[0]['coordinates'])
        for frame in range(self.frames):
            cluster_patch = []
            for robot_trace in self.cluster_trace:
                cluster_patch.append(RegularPolygon(robot_trace['coordinates'][frame], 3, radius=robot_trace['size'],
                                                    orientation=robot_trace['angle'][frame], color='k'))
            self.cluster_collections.append(cluster_patch)
        self.cluster = collections.PatchCollection(self.cluster_collections[0], match_original=True)
        self.ax.add_collection(self.cluster)

        # Drawing vision
        if self.robots_vision:
            self.scatter = plt.scatter(self.detected_points_trace[0][:, 0], self.detected_points_trace[0][:, 1],
                                       s=0.5, c='r')
            self.ax.add_collection(self.scatter)
          #  self.line = collections.LineCollection(self.detection_line_trace[0], colors='r')
          #  self.ax.add_collection(self.line)

        animation = FuncAnimation(self.fig, self.update, interval=interval, frames=2*self.frames, blit=False)
        # Show
        plt.show()

    def update(self, frame):
        if self.collision_frame is None:
            frame = frame % self.frames
        elif frame >= self.frames:
            frame = self.frames - 1
        self.cluster.set_paths(self.cluster_collections[frame])
        if self.robots_vision:
          #  self.line.set_paths(self.detection_line_trace[frame])
            self.scatter.set(offsets=self.detected_points_trace[frame])
        return

    def pause(self):
        pass

    def to_gif(self):
        pass
