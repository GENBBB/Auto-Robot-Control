from collections import defaultdict
from typing import List

import numpy as np
from scipy.spatial import ConvexHull, KDTree


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root != y_root:
            self.parent[y_root] = x_root

class Splitter:
    def __init__(self, splitter_settings: dict):
        self.convex = splitter_settings['convex']
        self.radius = splitter_settings['radius']
        self.merge_distance = splitter_settings['merge_distance']

    def obstacle_split(self, detected_points: np.ndarray, x: np.ndarray) -> List[List[np.ndarray]]:
        bool_matrix = np.linalg.norm(detected_points - x[:, np.newaxis, :], axis=2) < self.radius
        shift_bools = bool_matrix[:, 1:] != bool_matrix[:, :-1]
        indices = np.where(shift_bools)

        row_indices = indices[0]
        col_indices = indices[1] + 1

        split_results = [np.split(row, col_indices[row_indices == i]) for i, row in enumerate(detected_points)]

        bool_first = bool_matrix[:, 0]
        obstacles = [s[0::2] if bf else s[1::2] for s, bf in zip(split_results, bool_first)]

        for i in range(len(obstacles)):
            if len(obstacles[i]) > 1:
                if bool_matrix[i, 0] and bool_matrix[i, -1]:
                    obstacles[i][0] = np.vstack((obstacles[i][-1], obstacles[i][0]))
                    del obstacles[i][-1]

        if self.merge_distance > 0:
            obstacles = self.merge_all_groups(obstacles)
        if self.convex:
            obstacles = self.to_convex(obstacles)
        return obstacles

    @staticmethod
    def are_collinear(points: np.ndarray) -> bool:
        if len(points) < 3:
            return True

        p0, p1, *rest = points
        v1 = p1 - p0
        if np.allclose(v1, 0, atol=1e-8):
            return np.allclose(points, p0, atol=1e-8)

        normal = np.array([-v1[1], v1[0]])
        vectors = points - p0
        dots = np.dot(vectors, normal)
        return np.allclose(dots, 0, atol=1e-8)

    def compute_convex_hull(self, points: np.ndarray) -> np.ndarray:
        n = len(points)

        if n <= 2:
            return points.copy()

        if self.are_collinear(points):
            x = points[:, 0]
            y = points[:, 1]
            x_min, x_max = np.argmin(x), np.argmax(x)
            if x[x_min] != x[x_max]:
                return np.vstack([points[x_min], points[x_max]])
            else:
                y_min, y_max = np.argmin(y), np.argmax(y)
                return np.vstack([points[y_min], points[y_max]])

        hull = ConvexHull(points)
        return points[hull.vertices]

    def to_convex(self, data: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        return [[self.compute_convex_hull(arr) for arr in group] for group in data]

    def merge_single_group(self, groups: List[np.ndarray]) -> List[np.ndarray]:
        n = len(groups)
        if n == 0:
            return []

        extended_bboxes = []
        for arr in groups:
            if len(arr) == 0:
                extended_bboxes.append(None)
                continue
            min_x, max_x = np.min(arr[:, 0]), np.max(arr[:, 0])
            min_y, max_y = np.min(arr[:, 1]), np.max(arr[:, 1])
            extended_bboxes.append((
                min_x - self.merge_distance,
                max_x + self.merge_distance,
                min_y - self.merge_distance,
                max_y + self.merge_distance
            ))

        dsu = DSU(n)

        for i in range(n):
            for j in range(i + 1, n):
                if extended_bboxes[i] is None or extended_bboxes[j] is None:
                    continue

                bbox_i = extended_bboxes[i]
                bbox_j = extended_bboxes[j]

                overlap_x = bbox_i[0] <= bbox_j[1] and bbox_j[0] <= bbox_i[1]
                overlap_y = bbox_i[2] <= bbox_j[3] and bbox_j[2] <= bbox_i[3]
                if not (overlap_x and overlap_y):
                    continue

                arr_i = groups[i]
                arr_j = groups[j]
                if len(arr_i) == 0 or len(arr_j) == 0:
                    continue

                tree_j = KDTree(arr_j)
                min_dist = np.min([tree_j.query(point)[0] for point in arr_i])
                if min_dist < self.merge_distance:
                    dsu.union(i, j)

        clusters = defaultdict(list)
        for idx in range(n):
            root = dsu.find(idx)
            clusters[root].append(groups[idx])

        merged = []
        for cluster in clusters.values():
            merged.append(np.concatenate(cluster, axis=0))

        return merged

    def merge_all_groups(self, input_groups: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        return [self.merge_single_group(group) for group in input_groups]