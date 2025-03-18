import time

from config import config
import math

from diff_ev import ClusterDifferentialEvolution

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

def ex_1():
    experiment = {'n_obj': 20, 'min_size': min_size, 'max_size': max_size, 'n_robots': 1,
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': size_start_area, 'target_size': size_end_area, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area))]}

    return experiment

def ex_2():
    experiment = {'n_obj': 20, 'min_size': min_size, 'max_size': max_size, 'n_robots': 5,
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': size_start_area, 'target_size': size_end_area, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area))]}

    return experiment

def ex_3():
    experiment = {'n_obj': 20, 'min_size': min_size, 'max_size': max_size, 'n_robots': 15,
                  'area': (width, height), 'start': (x_start, y_start), 'target': (x_end, y_end),
                  'start_size': 20, 'target_size': 20, 'max_step': steps,
                  'excluded_area': [((x_start - size_start_area, y_start - size_start_area),
                                     (x_start + size_start_area, y_start + size_start_area))]}

    return experiment

def f():
    start_time = time.time()

    experiments = [ex_1(), ex_2(), ex_3()]

    lidar_settings = {'radius': radius, 'lidar_parts': lidar_parts}
    splitter_settings = {'convex': False, 'radius': radius, 'merge_distance': robot_size}

    static_settings = {'robot_size': robot_size, 'frequency': frequency, 'lidar_settings': lidar_settings,
                       'splitter_settings': splitter_settings, 'formation_distance': 3 * robot_size}

    bounds = {'gain': 10, 'min_letter': 1e-6, 'max_letter': 10, 'min_eps': 1e-4, 'max_eps': 1e-4,
              'min_h': 0.5, 'max_h': 0.5, 'min_gamma': 3, 'repulsive_distance_min': [2 * robot_size, robot_size],
              'radius': radius}

    evolution = ClusterDifferentialEvolution(static_settings, bounds, experiments, pop_size=50,
                                             max_generations=50, seed=seed, title='TO', cr=0.95, f=0.5)

    evolution.run()
    settings, fittness = evolution.get_current_best()
    evolution.animation_best()
    print('Params:')
    print(settings)
    print('Fitness:', fittness)
    print("Time: ", time.time() - start_time)


if __name__ == '__main__':
    f()