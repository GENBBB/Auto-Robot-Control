import time

import numpy as np

from animation import SystemAnimation
from config import config
import math

from diff_ev import ClusterDifferentialEvolution
from expanse import Area
from robot_system import Cluster

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
n_obj = int(config['Map']['n_obj'])
min_size = float(config['Map']['min_size'])
max_size = float(config['Map']['max_size'])
width = float(config['Config']['width'])
height = float(config['Config']['height'])
if seed == 'None':
    seed = None
else:
    seed = int(seed)

def st_g():
    experiment = {'n_obj': 25, 'min_size': min_size, 'max_size': max_size, 'n_robots': 1, 'type': 'static',
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': size_start_area, 'target_size': size_end_area, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area)),
                                    ((x_end - size_end_area, y_end - size_end_area),
                                     (x_end + size_end_area, y_end + size_end_area))
                                    ]}

    return experiment

def st():
    experiment = {'n_obj': 25, 'min_size': min_size, 'max_size': max_size, 'n_robots': 5, 'type': 'static',
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': size_start_area, 'target_size': size_end_area, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area)),
                                    ((x_end - size_end_area, y_end - size_end_area),
                                     (x_end + size_end_area, y_end + size_end_area))
                                    ]}

    return experiment

def dyn():
    experiment = {'n_obj': 10, 'min_size': min_size, 'max_size': max_size, 'n_robots': 5, 'type': 'dynamic',
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': size_start_area, 'target_size': size_end_area, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area)),
                                    ((x_end - size_end_area, y_end - size_end_area),
                                     (x_end + size_end_area, y_end + size_end_area))
                                    ]}

    return experiment

def letter():
    experiment = {'n_obj': 10, 'min_size': min_size, 'max_size': max_size, 'n_robots': 5, 'type': 'letter',
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': size_start_area, 'target_size': size_end_area, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area)),
                                    ((x_end - size_end_area, y_end - size_end_area),
                                     (x_end + size_end_area, y_end + size_end_area))
                                    ]}

    return experiment

def flock():
    experiment = {'n_obj': 0, 'min_size': min_size, 'max_size': max_size, 'n_robots': 10, 'type': 'static',
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': 30, 'target_size': 10, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area)),
                                    ((x_end - size_end_area, y_end - size_end_area),
                                     (x_end + size_end_area, y_end + size_end_area))
                                    ]}

    return experiment

def f():
    start_time = time.time()

    experiments = [st_g(), st()]

    lidar_settings = {'radius': radius, 'lidar_parts': lidar_parts}
    splitter_settings = {'convex': False, 'radius': radius, 'merge_distance': 1.5 * robot_size}

    static_settings = {'robot_size': robot_size, 'frequency': frequency, 'lidar_settings': lidar_settings,
                       'splitter_settings': splitter_settings, 'formation_distance': 3 * robot_size}

    bounds = {'min_alpha': [8.7, 2.92], 'max_alpha': [8.7, 2.92], 'min_beta': [18.36, 13.58], 'max_beta': [18.36, 13.58],
              'min_gamma': [6.99, 0.8], 'max_gamma': [6.99, 0.8], 'min_delta': [24.89, 10.64], 'max_delta': [24.89, 10.64],
              'min_a': 0.07, 'max_a': 0.07, 'min_b': 8.76, 'max_b': 8.76, 'min_eps': 0.16, 'max_eps': 0.16,
              'min_h': 0.18, 'max_h': 0.18, 'repulsive_distance_min': [3.62, 9.25], 'repulsive_distance_max': [3.62, 9.25],
              'radius': radius, 'v_min': 0, 'v_max': 10, 'g_min': [0, 0], 'g_max': [50, 1], 'p_min': [0], 'p_max': [2]}

    evolution = ClusterDifferentialEvolution(static_settings, bounds, experiments, pop_size=20,
                                             max_generations=10, seed=seed, title='DIPL2', cr=0.95, f=0.5)

    evolution.run()
    settings, fittness = evolution.get_current_best()
    evolution.animation_best()
    print('Params:')
    print(settings)
    print('Fitness:', fittness)
    print("Time: ", time.time() - start_time)


def best(mode):
    np.random.seed(seed)
    lidar_settings = {'radius': radius, 'lidar_parts': lidar_parts}
    splitter_settings = {'convex': False, 'radius': radius, 'merge_distance': 2 * robot_size}
    controller_settings = {'alpha': [8.7, 2.92], 'beta': [18.36, 13.58], 'gamma': [6.99, 2.1], 'delta': [24.89, 10.64],
                           'eps': 0.16, 'h': 0.18, 'a': 0.07, 'b': 8.76,
                           'formation_distance': 3.0, 'robot_distance': 3.9,
                           'object_distance': 6.25, 'formation_radius': 6.86,
                           'object_radius': 8.77, 'robot_radius': 3.62, 'v_max': 9.77, 'gyro': [66.56, 0.75], 'p_imp': 1.88}

    cluster = Cluster(n_robots, robot_size, frequency, lidar_settings, controller_settings, splitter_settings)
    area = Area()
    excluded_area = [((x_start - size_start_area, y_start - size_start_area),
                      (x_start + size_start_area, y_start + size_start_area)),
                     ((x_end - size_end_area, y_end - size_end_area),
                      (x_end + size_end_area, y_end + size_end_area))
                    ]
    area.random_static_set((width, height), n_obj, min_size, max_size, excluded_area, mode)
    arr = cluster.st_arrangement(n_robots, robot_size, np.array([x_start, y_start]), size_start_area)
    cluster.arrangement(arr[0], arr[1])
    target = np.array([x_end, y_end])
    for j in range(steps):
        cluster.update(area, target)
        area.update()
        if cluster.is_coming(target, size_end_area):
            break
    t = cluster.get_time()
    target_distance = cluster.get_target_distance()
    dead = cluster.get_dead()
    convexity = cluster.get_convexity()
    print(t, target_distance, dead, convexity)
    animation = SystemAnimation(area, cluster, True, robot_size,
                                width, height)
    animation.start()




if __name__ == '__main__':
    best('static')