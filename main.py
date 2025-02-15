import time

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
size_end_area = float(config['Cluster']['size_end_area'])
robot_size = float(config['Robot']['size'])
radius = float(config['Robot']['radius'])
lidar_parts = int(config['Robot']['lidar_parts'])
seed = config['Config']['seed']
if seed == 'None':
    seed = None
else:
    seed = int(seed)
gamma = [7.0, 4.0]
beta = [4.0, 0.0]
alpha = [0.5, 0.5]
delta = [50.0, 50.0]
rep = 1.0
h = 0.5
eps = 0.00001
a = 1
b = 10
distance = 6
distance_obj = 1.5
distance_rob = 3
convex = True

if __name__ == '__main__':
    start_time = time.time()
    map_obj = Area()
    np.random.seed(seed)
    map_obj.random_static_set()
 #   for i in range(20):
 #       map_obj.add(Circle(Point([10 + i * 5, 20 + i * 5]), 5))
 #       map_obj.add(Circle(Point([10 + i * 5, -5 + i * 5]), 5))

    lidar_settings = {'lidar_parts': lidar_parts, 'radius': radius}

    controller_settings = {'gamma': gamma, 'beta': beta, 'alpha': alpha, 'delta': delta, 'eps': eps, 'h': h, 'rep': rep,
                           'a': a, 'b': b, 'distance': distance, 'distance_obj': distance_obj,
                           'distance_rob': distance_rob, 'radius': radius}

    cluster = rob_sys.Cluster(n_robots, robot_size, frequency, lidar_settings, controller_settings)
    cluster.arrangement(Point([x_start, y_start]), size_start_area)
    collision_frame = None
    for i in range(steps):
        print("Step ", i)
        cluster.update(map_obj, Point([x_end, y_end]))
        if cluster.is_collision(map_obj) or cluster.is_coming(Point([x_end, y_end]), size_end_area):
            break
    print("Time: ", time.time() - start_time)

    animation = SystemAnimation(map_obj, cluster, robot_vision=True, size=robot_size)
    animation.start()
