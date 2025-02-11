import numpy as np

from config import config

max_dv = float(config['Robot']['max_dv'])

class Controller:
    def __init__(self, controller_settings: dict):
        self.gamma = controller_settings['gamma']
        self.beta = controller_settings['beta']
        self.alpha = controller_settings['alpha']
        self.delta = controller_settings['delta']
        self.rep = controller_settings['rep']
        self.h = controller_settings['h']
        self.eps = controller_settings['eps']
        self.alpha1 = controller_settings['alpha1']
        self.b = controller_settings['b']
        self.distance = controller_settings['distance']
        self.distance_rob = controller_settings['distance_rob']
        self.distance_obj = controller_settings['distance_obj']
        self.convex = controller_settings['convex']
        self.radius = controller_settings['radius']

    @staticmethod
    def to_convex(points):
        return points

    @staticmethod
    def gamma_function(z: float) -> float:
        return z / np.sqrt(1 + z ** 2)

    def bump_function(self, z: np.ndarray) -> np.ndarray:
        res = np.zeros_like(z)

        mask = (z >= 0) & (z < self.h)
        res[mask] = 1

        mask = (z >= self.h) & (z <= 1)
        res[mask] = 0.5 * (1 + np.cos(np.pi * ((z[mask] - self.h) / (1 - self.h))))
        return res

    def sigma_norm(self, z: np.ndarray) -> np.ndarray:
        return (np.sqrt(1 + self.eps * z ** 2) - 1) / self.eps

    def repulsive_function(self, z: float, d) -> float:
        return self.bump_function(z / self.sigma_norm(d)) * (self.gamma_function(z - self.sigma_norm(d)) - 1)

    def alpha_function(self, z: np.ndarray) -> np.ndarray:
        return 0.5 * ((self.alpha1 + self.b) * self.gamma_function(z + (self.b - self.alpha1) / np.sqrt(4 * self.alpha1 * self.b)) + self.alpha1 - self.b)

    def attractive_functions(self, z: np.ndarray) -> np.ndarray:
        return self.bump_function(z / self.sigma_norm(self.radius)) * self.alpha_function(z - self.sigma_norm(self.distance))

    def alpha_control(self, x, v):
        diff_q = np.full((len(x), len(x), 2), x)
        diff_q = diff_q - np.transpose(diff_q, axes=(1, 0, 2))

        diff_p = np.full((len(v), len(v), 2), v)
        diff_p = diff_p - np.transpose(diff_p, axes=(1, 0, 2))

        norm = np.linalg.norm(diff_q, axis=2)

        u_position = (self.attractive_functions(self.sigma_norm(norm)) * diff_q.transpose((2, 0 ,1)) /
                      np.sqrt(1 + self.eps * norm ** 2))

        u_velocity = (self.bump_function(self.sigma_norm(norm) / self.sigma_norm(self.radius)) *
                      diff_p.transpose((2, 0, 1)))

        u_position = np.sum(u_position, axis=2)
        u_velocity = np.sum(u_velocity, axis=2)

        u = self.alpha[0] * u_position + self.alpha[1] * u_velocity
        return u.transpose((1, 0))

    def beta_control(self, x):
        return np.zeros((len(x), 2))
        sum1 = np.zeros(2)
        sum2 = np.zeros(2)
        for point in detected_points:
            mu = 0.1 / np.linalg.norm(self.position - point)
            q = mu * self.position + (1 - mu) * point
            diff = q - self.position
            a = np.reshape((self.position - point) / np.linalg.norm(self.position - point), (2, 1))
            p = mu * np.matmul((np.eye(2) - np.matmul(a, np.transpose(a))), self.v)
            norm = np.linalg.norm(diff)
            sum1 += self.repulsive_function(self.sigma_norm(norm), distance_obj) * diff / np.sqrt(1 + eps * norm ** 2)
            sum2 += self.bump_function(self.sigma_norm(norm) / self.sigma_norm(distance_obj)) * (p - self.v)
        return beta_const[0] * sum1 + beta_const[1] * sum2

    def gamma_control(self, x, v, end_area):
        diff = x - end_area
        return -self.gamma[0] * diff / np.sqrt(1 + np.linalg.norm(diff)) - self.gamma[1] * v

    def delta_control(self, cluster):
        sum1 = np.zeros(2)
        sum2 = np.zeros(2)
        for robot in cluster:
            diff = robot.position - self.position
            norm = np.linalg.norm(diff)
            sum1 += self.repulsive_function(self.sigma_norm(norm), self.distance_rob) * diff / np.sqrt(1 + eps * norm ** 2)
            sum2 += self.bump_function(self.sigma_norm(norm) / self.sigma_norm(self.distance_rob)) * (robot.v - self.v)
        return delta_const[0] * sum1 + delta_const[1] * sum2

    def control(self, x, v, detected_points, target):
        u = np.zeros_like(x)
        u += self.alpha_control(x, v)
        u += self.gamma_control(x, v, target)
        return u
