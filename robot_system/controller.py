from typing import List

import numpy as np


class Controller:
    def __init__(self, controller_settings: dict):
        self.alpha = controller_settings['alpha']
        self.beta = controller_settings['beta']
        self.gamma = controller_settings['gamma']
        self.delta = controller_settings['delta']

        self.h = controller_settings['h']
        self.eps = controller_settings['eps']

        self.a = controller_settings['a']
        self.b = controller_settings['b']
    #    if self.a > self.b:
    #        raise ValueError('a must be smaller or equal than b')
        self.c = self.b - self.a / np.sqrt(4 * self.a * self.b)
        self.alpha_sum = self.a + self.b
        self.alpha_diff = self.a - self.b

        self.formation_distance = controller_settings['formation_distance']
        self.distance_rob = controller_settings['robot_distance']
        self.distance_obj = controller_settings['object_distance']
        self.formation_radius = controller_settings['formation_radius']
        self.robot_radius = controller_settings['robot_radius']
        self.object_radius = controller_settings['object_radius']

        self.d_obj = self.sigma_norm(self.distance_obj)
        self.d_rob = self.sigma_norm(self.distance_rob)
        self.d_f = self.sigma_norm(self.formation_distance)
        self.r_obj = self.sigma_norm(self.object_radius)
        self.r_rob = self.sigma_norm(self.robot_radius)
        self.r_f = self.sigma_norm(self.formation_radius)

    @staticmethod
    def gamma_function(z: np.ndarray) -> np.ndarray:
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

    def repulsive_function_obj(self, z: np.ndarray) -> np.ndarray:
        return self.bump_function(z / self.r_obj) * (self.gamma_function(z - self.d_obj) - 1)

    def repulsive_function_rob(self, z: np.ndarray) -> np.ndarray:
        return self.bump_function(z / self.r_rob) * (self.gamma_function(z - self.d_rob) - 1)

    def alpha_function(self, z: np.ndarray) -> np.ndarray:
        return 0.5 * (self.alpha_sum * self.gamma_function(z + self.c) + self.alpha_diff)

    def attractive_functions(self, z: np.ndarray) -> np.ndarray:
        return self.bump_function(z / self.r_f) * self.alpha_function(z - self.d_f)

    def alpha_control(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        diff_q = np.full((len(x), len(x), 2), x)
        diff_q = diff_q - np.transpose(diff_q, axes=(1, 0, 2))

        diff_p = np.full((len(v), len(v), 2), v)
        diff_p = diff_p - np.transpose(diff_p, axes=(1, 0, 2))

        norm = np.linalg.norm(diff_q, axis=2)

        u_position = (self.attractive_functions(self.sigma_norm(norm)) * diff_q.transpose((2, 0 ,1)) /
                      np.sqrt(1 + self.eps * norm ** 2))
        u_position_delta = self.repulsive_function_rob(self.sigma_norm(norm)) * diff_q.transpose((2, 0, 1)) / np.sqrt(
            1 + self.eps * norm ** 2)


        u_velocity = self.bump_function(self.sigma_norm(norm) / self.r_f) * diff_p.transpose((2, 0, 1))
        u_velocity_delta = self.bump_function(self.sigma_norm(norm) / self.r_rob) * diff_p.transpose((2, 0, 1))

        u_position = np.sum(u_position, axis=2)
        u_velocity = np.sum(u_velocity, axis=2)
        u_velocity_delta = np.sum(u_velocity_delta, axis=2)
        u_position_delta = np.sum(u_position_delta, axis=2)

        u = self.alpha[0] * u_position + self.alpha[1] * u_velocity
        u += self.delta[0] * u_position_delta + self.delta[1] * u_velocity_delta
        return u.transpose((1, 0))

    @staticmethod
    def compute_unit_normals(vectors: np.ndarray) -> np.ndarray:
        x = vectors[..., 0]
        y = vectors[..., 1]

        normals = np.stack((-y, x), axis=-1)

        norms = np.linalg.norm(normals, axis=-1, keepdims=True)

        non_zero_mask = (norms.squeeze(-1) != 0) & np.isfinite(norms.squeeze(-1))
        unit_normals = np.zeros_like(vectors)
        unit_normals[non_zero_mask] = normals[non_zero_mask] / norms[non_zero_mask]
        return unit_normals

    @staticmethod
    def compute_projection_matrices(normals: np.ndarray) -> np.ndarray:
        identity = np.zeros(normals.shape + (2,))
        identity[..., 0, 0] = 1
        identity[..., 1, 1] = 1

        outer = np.einsum('...i,...j->...ij', normals, normals)
        projection_matrices = identity - outer
        return projection_matrices

    @staticmethod
    def project_vectors(projection_mats: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        vectors_expanded = vectors[:, np.newaxis, :]

        result = np.einsum('n m i j, n m j -> n m i', projection_mats, vectors_expanded)
        return result

    def beta_control(self, x : np.ndarray, v: np.ndarray, beta_position: List[np.ndarray],
                     beta_direction: List[np.ndarray]) -> np.ndarray:
        n = len(beta_position)
        max_neighbors = max(len(neighbors) for neighbors in beta_position)


        closest = np.full((n, max_neighbors, 2), np.inf)
        normal = np.full((n, max_neighbors, 2), np.inf)
        for i in range(n):
            m = len(beta_position[i])
            if m > 0:
                closest[i, :m] = beta_position[i]
                normal[i, :m] = beta_direction[i]
        normal = self.compute_unit_normals(normal)
        projection_matrices = self.compute_projection_matrices(normal)
        beta_p = self.project_vectors(projection_matrices, v)


        diff_q = closest - x[:, np.newaxis, :]
        diff_q[closest == np.inf] = 0
        norm = np.linalg.norm(diff_q, axis=2)
        u_position = self.repulsive_function_obj(self.sigma_norm(norm)) * diff_q.transpose((2, 0, 1)) / np.sqrt(1 + self.eps * norm ** 2)


        diff_p = beta_p - v[:, np.newaxis, :]
        diff_p[closest == np.inf] = 0
        u_velocity = self.bump_function(self.sigma_norm(norm) / self.r_obj) * diff_p.transpose((2, 0, 1))

        u_position = np.sum(u_position, axis=2)
        u_velocity = np.sum(u_velocity, axis=2)
        u = self.beta[0] * u_position + self.beta[1] * u_velocity
        return u.transpose((1,0))

    def gamma_control(self, x: np.ndarray, v: np.ndarray, end_area: np.ndarray) -> np.ndarray:
        diff = x - end_area
        return -self.gamma[0] * diff / np.sqrt(1 + np.linalg.norm(diff)) - self.gamma[1] * v

    def gyro_control(self, beta_control: np.ndarray) -> np.ndarray:
        pass

    def control(self, x: np.ndarray, v: np.ndarray, beta_position: List[np.ndarray], beta_direction: List[np.ndarray],
                target: np.ndarray) -> np.ndarray:
        u = np.zeros_like(x)
        u += self.alpha_control(x, v)
        u += self.beta_control(x, v, beta_position, beta_direction)
        u += self.gamma_control(x, v, target)
        return u