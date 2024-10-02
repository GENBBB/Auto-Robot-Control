import robot_system as rob_sys
from expanse.area import Area
from expanse.obstacles import Circle
from animation import SystemAnimation
from skspatial.objects import Point
from config import config
import math
import numpy as np
from collections import Counter

frequency = int(config['Config']['frequency'])
steps = math.ceil(float(config['Config']['duration']) * frequency)
n_robots = int(config['Cluster']['n_rob'])
x_start = float(config['Cluster']['x_start'])
y_start = float(config['Cluster']['y_start'])
x_end = float(config['Cluster']['x_end'])
y_end = float(config['Cluster']['y_end'])
size_start_area = float(config['Cluster']['size_start_area'])
robot_size = float(config['Robot']['size'])
radius = float(config['Robot']['radius'])
lidar_parts = int(config['Robot']['lidar_parts'])
seed = config['Config']['seed']
if seed == 'None':
    seed = None
else:
    seed = int(seed)

if __name__ == '__main__':
    circle = Circle(Point([10, 10]), 5)
    map_obj = Area([circle])
    cluster = rob_sys.create_cluster(1, Point([2, 2]), 0, radius, robot_size,
                                     lidar_parts, frequency, Point([x_end, y_end]))
    collision_frame = None
    cluster.update(map_obj)
    cluster.update(map_obj)
    cluster.update(map_obj)
    animation = SystemAnimation(map_obj, cluster, 5, 2, robot_vision=True,)
    animation.start()