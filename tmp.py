import numpy as np
import matplotlib.pyplot as plt


def find_intersections(points, circles, n_rays):
    points = np.array(points)
    num_points = points.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))

    intersections = np.full((num_points, n_rays, 2), np.inf)  # Заполняем бесконечными значениями

    for cx, cy, r in circles:
        oc = np.array([cx, cy]) - points[:, None, :]
        proj = np.einsum('ijk,jk->ij', oc, directions)
        oc_norm2 = np.sum(oc ** 2, axis=2)
        disc = proj ** 2 - oc_norm2 + r ** 2

        valid = disc >= 0
        sqrt_disc = np.sqrt(np.maximum(disc, 0))

        t1 = proj - sqrt_disc
        t2 = proj + sqrt_disc
        t_min = np.where((t1 > 0) & valid, t1, np.where((t2 > 0) & valid, t2, np.inf))

        mask = t_min < np.linalg.norm(intersections - points[:, None, :], axis=2)
        intersections[mask, :] = (points[:, None, :] + t_min[..., None] * directions)[mask, :]

    return intersections


def plot_results(points, circles, intersections, n_rays):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for cx, cy, r in circles:
        circle = plt.Circle((cx, cy), r, color='blue', fill=False)
        ax.add_patch(circle)

    points = np.array(points)
    intersections = np.array(intersections)

    for i, point in enumerate(points):
        ax.scatter(*point, color='red', marker='o')
        for j in range(n_rays):
            inter = intersections[i, j]
            if not np.isinf(inter).any():
                ax.plot([point[0], inter[0]], [point[1], inter[1]], color='black', linestyle='dashed')
                ax.scatter(*inter, color='green', marker='x')

    ax.set_xlim(-5, 10)
    ax.set_ylim(-5, 10)
    plt.show()


# Пример использования
points = [(0, 0), (2, 2)]
circles = [(3, 3, 2), (-2, 4, 1.5), (5, 0, 3)]
n_rays = 8
intersections = find_intersections(points, circles, n_rays)
print(intersections)
plot_results(points, circles, intersections, n_rays)