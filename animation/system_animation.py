

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.collections as collections
from matplotlib.patches import RegularPolygon
import matplotlib

from expanse.area import Area
from robot_system import Cluster
import numpy as np
from typing import Self

interval = 15

class SystemAnimation:
    """
    Class for animating the movement of a cluster in an area with obstacles
    """
    def __init__(self, area: Area, cluster: Cluster, robot_vision: bool = False, size: float = 1,
                 width: float = 100, height: float = 100):
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
        self.circle_trace, self.circle_size, _ = area.obstacles.parse(30)
        self.cluster_trace, self.angles, self.detected_points_trace, self.beta_track, self.frames = cluster.parse_trace(30)

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
        self.width = width
        self.height = height
        self.circle = None

    def start(self) -> Self:
        """
        Running cluster motion animation
        Returns
        -------

        """
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlim=[0, self.width], ylim=[0, self.height], xticks=range(0, int(self.width), int(self.width) // 20),
                    yticks=range(0, int(self.height), int(self.height) // 20))
        # Drawing map

        self.circle = collections.EllipseCollection(2 * self.circle_size, 2 * self.circle_size,
                                                                          angles=0, units='xy',
                                                                          offsets=self.circle_trace[0],
                                                                          offset_transform=self.ax.transData)
        self.ax.add_collection(self.circle)

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
                if frame % 3 == 0:
                    self.trace[frame].append(self.cluster_trace[frame][i])
            if frame != self.frames - 1:
                self.trace[frame+1] = self.trace[frame]
            self.trace[frame] = np.array(self.trace[frame])
        if False:
            self.scatter_trace = plt.scatter(self.trace[0][:, 0], self.trace[0][:, 1], s = 0.5, c='b', zorder=0)
            self.ax.add_collection(self.scatter_trace)


        if self.robots_vision and False:
            self.scatter = plt.scatter(self.detected_points_trace[0][:, 0], self.detected_points_trace[0][:, 1],
                                       s=0.5, c='r')
            self.ax.add_collection(self.scatter)
            self.beta = plt.scatter(self.beta_track[0][:, 0], self.beta_track[0][:, 1], s=5, c='b')
            self.ax.add_collection(self.beta)
          #  self.line = collections.LineCollection(self.detection_line_trace[0], colors='r')
          #  self.ax.add_collection(self.line)


        animation = FuncAnimation(self.fig, self.update, interval=interval, frames=self.frames, blit=False, repeat=True)
        matplotlib.rcParams['animation.ffmpeg_path'] = "ffmpeg\\bin\\ffmpeg.exe"
        writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        tmp = np.random.randint(0, 1000)
        filename = str(tmp) + '_pres_dipl_tmp.mp4'
        animation.save(filename, writer=writer)
       # plt.grid()
        plt.show()
        return

    def update(self, frame):
        frame = frame % self.frames
       # self.scatter_trace.set(offsets=self.trace[frame])
        if self.robots_vision and False:
            pass
           # self.scatter.set(offsets=self.detected_points_trace[frame])
            #self.beta.set(offsets=self.beta_track[frame])
          #  self.line.set_paths(self.detection_line_trace[frame])
        self.circle.set_offsets(self.circle_trace[frame])
        self.cluster.set_paths(self.cluster_collections[frame])
        return self.cluster

    def pause(self):
        pass

    def to_gif(self):
        pass
