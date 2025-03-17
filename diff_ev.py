from typing import List, Tuple

import numpy as np
import pandas as pd

from expanse import Area
from robot_system import Cluster
from animation import SystemAnimation


class ClusterDifferentialEvolution:
    def __init__(self, static_settings: dict, bounds: dict, experiments: List, pop_size: int = 50, max_generations: int = 200,
                 f: float = 0.8, cr: float = 0.9, seed: int or None = None, title: str = "UNKNOWN"):
        np.random.seed(seed)

        self.seed = seed
        self.title = title

        self.bounds = bounds
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.F = f
        self.CR = cr
        self.dimensions = 17

        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')

        columns = ['generation', 'params', 'T', 'target_distance', 'dead', 'convexity']
        self.df = pd.DataFrame(columns=columns)

        self.experiments = experiments
        self.static_settings = static_settings
        self.areas = []
        self.arrangements = [Cluster.st_arrangement(experiment['n_robots'], static_settings['robot_size'],
                                                    experiment['start'],
                                                    experiment['start_size']) for experiment in experiments]

        for experiment in self.experiments:
            area = Area()
            area.random_static_set(experiment['area'], experiment['n_obj'], experiment['min_size'],
                                   experiment['max_size'],
                                   experiment['excluded_area'])
            self.areas.append(area)

    def _param_to_dict(self, param):
        setting = {'alpha': param[0:2],
                   'beta': param[2:4],
                   'gamma': param[4:6],
                   'delta': param[6:8],
                   'h': param[8],
                   'eps': param[9],
                   'a': param[10],
                   'b': param[11],
                   'formation_radius': param[12],
                   'robot_radius': param[13],
                   'object_radius': param[14],
                   'formation_distance': self.static_settings['formation_distance'],
                   'robot_distance': param[15],
                   'object_distance': param[16]}
        return setting

    def add_to_df(self, generation: int, param: np.ndarray, time: np.ndarray, target_distance: np.ndarray,
                  dead: np.ndarray, convexity: np.ndarray) -> None:
        new_row = {
            'generation': [generation],
            'params': [param.tolist()],
            'T': [time.tolist()],
            'target_distance': [target_distance.tolist()],
            'dead': [dead.tolist()],
            'convexity': [convexity.tolist()],
        }
        self.df = pd.concat([self.df, pd.DataFrame(new_row)], ignore_index=True)

    @staticmethod
    def score(time: np.ndarray, target_distance: np.ndarray, dead: np.ndarray, convexity: np.ndarray) -> float:
        score = np.mean(target_distance, dtype=np.float64)
        score += np.mean(time, dtype=np.float64)
        score += np.mean(dead, dtype=np.float64) * 2000
        print(score)
        return score

    def _map(self, setting: dict, area: Area, experiment: dict, arrangement: Tuple,
             animation_flag: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cluster = Cluster(experiment['n_robots'], self.static_settings['robot_size'],
                          self.static_settings['frequency'], self.static_settings['lidar_settings'],
                          setting, self.static_settings['splitter_settings'])
        cluster.arrangement(np.copy(arrangement[0]), np.copy(arrangement[1]))
        for j in range(experiment['max_step']):
            cluster.update(area, experiment['target'])
            if cluster.is_coming(experiment['target'], experiment['target_size']):
                break
        time = cluster.get_time()
        target_distance = cluster.get_target_distance()
        dead = cluster.get_dead()
        convexity = cluster.get_convexity()
        print(time, target_distance, dead, convexity)
        if animation_flag:
            animation = SystemAnimation(area, cluster, True, self.static_settings['robot_size'])
            animation.start()
        return time, target_distance, dead, convexity

    def _race(self, param: np.ndarray, version: int = 0, generation: int = -1) -> float:
        setting = self._param_to_dict(param)
        time = np.empty(len(self.experiments))
        target_distance = np.empty(len(self.experiments))
        dead = np.empty(len(self.experiments))
        convexity = np.empty(len(self.experiments))
        print("Version #", version)
        for i in range(len(self.experiments)):
            result = self._map(setting, self.areas[i], experiment=self.experiments[i], arrangement=self.arrangements[i])
            time[i], target_distance[i], dead[i], convexity[i] = result
        self.add_to_df(generation, param, time, target_distance, dead, convexity)
        return self.score(time, target_distance, dead, convexity)

    def _random_param(self):
        gain = np.random.uniform(0, self.bounds['gain'], 4)
        gamma = np.random.uniform(self.bounds['min_gamma'], self.bounds['gain'], 2)
        delta = np.random.uniform(0, self.bounds['gain'], 2)
        h = np.random.uniform(self.bounds['min_h'], self.bounds['max_h'], 1)
        eps = np.random.uniform(self.bounds['min_eps'], self.bounds['max_eps'], 1)
        letter = np.random.uniform(self.bounds['min_letter'], self.bounds['max_letter'], 2)
        formation_radius = np.random.uniform(self.static_settings['formation_distance'], self.bounds['radius'], 1)
        repulsive_radius = np.random.uniform(self.bounds['repulsive_distance_min'], self.bounds['radius'], 2)
        repulsive_distance = np.random.uniform(self.bounds['repulsive_distance_min'], repulsive_radius, 2)
        param = np.concatenate((gain, gamma, delta, h, eps, letter, formation_radius, repulsive_radius, repulsive_distance))
        return param

    def _initialize_population(self):
        self.population = [self._random_param() for _ in range(self.pop_size)]
        self.population = np.array(self.population)
        self.fitness = [self._race(self.population[i], i, -1) for i in range(len(self.population))]

    def _mutation(self, idx):
        """Стратегия мутации DE/rand/1"""
        candidates = [i for i in range(self.pop_size) if i != idx]
        a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
        return a + self.F * (b - c)

    def _crossover(self, mutant, target_idx):
        """Бинарный кроссовер"""
        cross_points = np.random.rand(self.dimensions) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def _clip_within_bounds(self, individual):
        """Ограничение значений переменных в заданных границах"""
        individual[:4] = np.clip(individual[:4], 0, self.bounds['gain'])
        individual[4:6] = np.clip(individual[4:6], self.bounds['min_gamma'], self.bounds['gain'])
        individual[6:8] = np.clip(individual[6:8], 0, self.bounds['gain'])
        individual[8] = np.clip(individual[8], self.bounds['min_h'], self.bounds['max_h'])
        individual[9] = np.clip(individual[9], self.bounds['min_eps'], self.bounds['max_eps'])
        individual[10:12] = np.clip(individual[10:12], self.bounds['min_letter'], self.bounds['max_letter'])
        individual[12] = np.clip(individual[12], self.static_settings['formation_distance'], self.bounds['radius'])
        individual[13:15] = np.clip(individual[13:15], self.bounds['repulsive_distance_min'], self.bounds['radius'])
        individual[15:17] = np.clip(individual[15:17], self.bounds['repulsive_distance_min'], individual[13:15])
        return individual

    def save_df(self):
        filename = 'experiments/' + self.title + '_' + str(self.seed) + '.csv'
        self.df.to_csv(filename, index=False)

    def run(self):
        """Запуск алгоритма оптимизации"""
        self._initialize_population()
        self.best_fitness = np.min(self.fitness)
        self.best_solution = self.population[np.argmin(self.fitness)]

        for generation in range(self.max_generations):
            for i in range(self.pop_size):
                # Генерация пробного решения
                mutant = self._mutation(i)
                mutant = self._clip_within_bounds(mutant)
                trial = self._crossover(mutant, i)

                # Оценка качества
                trial_fitness = self._race(trial, i, generation)

                # Жадный отбор
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    # Обновление лучшего решения
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial.copy()

            # Визуализация прогресса (опционально)
            if (generation % 1) == 0:
                print("#####")
                print(f"Generation {generation}: Best Fitness = {self.best_fitness:.6f}")
                print("#####")

        self.save_df()

        return self.best_solution, self.best_fitness

    def get_current_best(self):
        """Возвращает текущее лучшее решение и значение функции"""
        return self._param_to_dict(self.best_solution), self.best_fitness

    def animation_best(self):
        setting = self._param_to_dict(self.best_solution)
        time = np.empty(len(self.experiments))
        target_distance = np.empty(len(self.experiments))
        dead = np.empty(len(self.experiments))
        convexity = np.empty(len(self.experiments))
        for i in range(len(self.experiments)):
            result = self._map(setting, self.areas[i], experiment=self.experiments[i],
                               arrangement=self.arrangements[i], animation_flag=True)
            time[i], target_distance[i], dead[i], convexity[i] = result