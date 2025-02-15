import numpy as np
import matplotlib.pyplot as plt


def find_intersections(points, circles, n_rays, turns):
    points = np.array(points)  # (num_points, 2)
    circles = np.array(circles)  # (num_circles, 3)
    num_points = points.shape[0]

    # Генерация направлений лучей с учетом поворота
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)  # (n_rays,)
    turns = np.array(turns)[:, None]  # (num_points, 1)
    rotated_angles = angles + turns  # (num_points, n_rays)

    directions = np.stack((np.cos(rotated_angles), np.sin(rotated_angles)), axis=-1)  # (num_points, n_rays, 2)

    # Векторизованный расчет расстояний от точек до центров окружностей
    oc = circles[:, None, None, :2] - points[None, :, None, :]  # (num_circles, num_points, n_rays, 2)
    print(oc.shape)
    print(directions.shape)
    directions = np.tile(directions, (circles.shape[0], 1, 1, 1))
    print(directions.shape)
    proj = np.einsum('ijkl,ijkl->ijk', oc, directions)  # (num_circles, num_points, n_rays)

    oc_norm2 = np.sum(oc ** 2, axis=3)  # (num_circles, num_points, n_rays)
    r2 = circles[:, None, None, 2] ** 2  # (num_circles, 1, 1)
    print(r2.shape)
    disc = proj ** 2 - oc_norm2 + r2  # (num_circles, num_points, n_rays)

    valid = disc >= 0  # Проверяем, есть ли пересечение
    sqrt_disc = np.sqrt(np.maximum(disc, 0))  # Берем корень из дискриминанта

    t1 = proj - sqrt_disc
    t2 = proj + sqrt_disc

    min_t = np.minimum(t1, t2)
    min_t[~valid] = np.inf  # Убираем лучи, не пересекающие окружность

    best_t = np.min(min_t, axis=0)  # Берем минимальное t среди всех окружностей
    intersections = points[:, None, :] + best_t[..., None] * directions  # Вычисляем пересечения

    return intersections

def plot_results(points, circles, intersections, n_rays):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for cx, cy, r in circles:
        circle = plt.Circle((cx, cy), r, color='blue', fill=False)
        ax.add_patch(circle)

    points = np.array(points)
    intersections = np.array(intersections)
    n_circles = intersections.shape[0]

    for i, point in enumerate(points):
        ax.scatter(*point, color='red', marker='o')
        for k in range(n_circles):
            for j in range(n_rays):
                inter = intersections[k, i, j]
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
turns = [0, 0]  # Углы поворота для каждой точки
intersections = find_intersections(points, circles, n_rays, turns)
plot_results(points, circles, intersections, n_rays)