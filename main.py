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
    map_obj = Area()
    np.random.seed(seed)
    map_obj.random_static_set()
    cluster = rob_sys.create_cluster(n_robots, Point([x_start, y_start]), size_start_area, radius, robot_size,
                                     lidar_parts, frequency, Point([x_end, y_end]))
    collision_frame = None
    for i in range(steps):
        print(i)
        try:
            cluster.update(map_obj)
        except RuntimeError:
            collision_frame = i
            print("zopa", i)
            break
    frames = range(0, steps, frequency // 30)
    animation = SystemAnimation(map_obj, cluster, len(frames), collision_frame, robot_vision=True,)
    animation.start()
