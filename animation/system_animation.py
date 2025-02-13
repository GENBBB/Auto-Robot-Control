


import matplotlib.pyplot as plt
from PIL.ImageChops import offset
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import matplotlib.collections as collections
from matplotlib.patches import RegularPolygon
from matplotlib.pyplot import scatter
from scipy.constants import point

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
    def __init__(self, area: Area, cluster: Cluster, robot_vision: bool = False, size: float = 1):
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
        self.cluster_trace, self.angles, self.detected_points_trace, self.frames = cluster.parse_trace(30)
        self.cluster_collections = None
        self.cluster = None
        self.fig = None
        self.ax = None
        self.scatter = None
        self.scatter_trace = None
        self.trace = None
        self.line_collection = None
        self.line = None
        self.robots_vision = robot_vision
        if not self.robots_vision:
            self.detected_points_trace = None
        self.size = size

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
        self.frames = len(self.cluster_trace)
        for frame in range(self.frames):
            cluster_patch = []
            for i in range(len(self.cluster_trace[0])):
                cluster_patch.append(RegularPolygon(self.cluster_trace[frame][i], 3, radius=self.size,
                                                    orientation=self.angles[frame][i], color='k'))
            self.cluster_collections.append(cluster_patch)
        self.cluster = collections.PatchCollection(self.cluster_collections[0], match_original=True, zorder=2)
        self.ax.add_collection(self.cluster)

        #Drawing trace
        self.trace = [[] for i in range(self.frames)]
        for frame in range(self.frames):
            for i in range(len(self.cluster_trace[0])):
                if frame % 2 == 0:
                    self.trace[frame].append(self.cluster_trace[frame][i])
            if frame != self.frames - 1:
                self.trace[frame+1] = self.trace[frame]
            self.trace[frame] = np.array(self.trace[frame])
        if False:
            self.scatter_trace = plt.scatter(self.trace[0][:, 0], self.trace[0][:, 1], s = 0.5, c='b', zorder=0)
            self.ax.add_collection(self.scatter_trace)


        if self.robots_vision:
            self.scatter = plt.scatter(self.detected_points_trace[0][:, 0], self.detected_points_trace[0][:, 1],
                                       s=0.5, c='r')
            self.ax.add_collection(self.scatter)
          #  self.line = collections.LineCollection(self.detection_line_trace[0], colors='r')
          #  self.ax.add_collection(self.line)

        animation = FuncAnimation(self.fig, self.update, interval=interval, frames=self.frames, blit=False, repeat=False)
        writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        tmp = np.random.randint(0, 1000)
        filename = str(tmp) + '.mp4'
#        animation.save(filename, writer=writer)
       # plt.grid()
        plt.show()
        return

    def update(self, frame):
 #       self.scatter_trace.set(offsets=self.trace[frame])
        if self.robots_vision:
            self.scatter.set(offsets=self.detected_points_trace[frame])
          #  self.line.set_paths(self.detection_line_trace[frame])
        self.cluster.set_paths(self.cluster_collections[frame])
        return self.cluster

    def pause(self):
        pass

    def to_gif(self):
        pass
