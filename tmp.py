import random
import math
import matplotlib.pyplot as plt


def dist(a, b):
    """Евклидово расстояние между точками a и b."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def circle_intersections(c1, c2, r):
    """
    Пересечение двух кругов радиуса r с центрами c1 и c2.
    Возвращает список из 0, 1 или 2 точек.
    """
    x0, y0 = c1
    x1, y1 = c2
    d = dist(c1, c2)
    if d < 1e-6 or d > 2*r:
        return []
    # расстояние от c1 до основания хорды пересечения
    a = d / 2
    h = math.sqrt(max(r*r - a*a, 0.0))
    # точка середины хорды
    xm = x0 + a*(x1 - x0)/d
    ym = y0 + a*(y1 - y0)/d
    # вектор, перпендикулярный c1->c2
    rx = -(y1 - y0) * (h/d)
    ry =  (x1 - x0) * (h/d)
    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    return [p1] if h < 1e-6 else [p1, p2]


def valid(candidate, pts, d, r, tol=1e-6):
    """
    Проверка, что candidate не нарушает правила:
      для каждой уже стоящей точки p:
        если dist(candidate, p) < r, то |dist - d| < tol
    """
    for p in pts:
        di = dist(candidate, p)
        if di < tol:
            return False
        if di < r - tol and abs(di - d) > tol:
            return False
    return True


def build_points(N=5, d=0.5, r=1.0, r_max=2.0, seed=None):
    """
    Итеративно строит N точек по вашему алгоритму с рандомизацией порядка шагов:
    - 90%: две → одна → далеко;
    -  5%: одна → далеко → две;
    -  5%: далеко → две → одна.
    """
    if seed is not None:
        random.seed(seed)

    pts = [(0.0, 0.0), (0.0, d)]  # стартовые точки

    while len(pts) < N:
        constructed = False
        # выбираем последовательность шагов
        r_choice = random.random()
        if r_choice < 0.80:
            sequence = ['two', 'one', 'none']
        elif r_choice < 0.90:
            sequence = ['one', 'none', 'two']
        else:
            sequence = ['none', 'two', 'one']

        # выполняем шаги в выбранном порядке
        for step in sequence:
            if constructed:
                break
            if step == 'two':
                # пробуем строить по двум точкам
                inds = list(range(len(pts)))
                random.shuffle(inds)
                for i in inds:
                    for j in inds:
                        if i >= j:
                            continue
                        for c in circle_intersections(pts[i], pts[j], d):
                            if valid(c, pts, d, r):
                                pts.append(c)
                                constructed = True
                                break
                        if constructed:
                            break
                    if constructed:
                        break

            elif step == 'one':
                # пробуем строить по одной точке
                for _ in range(100):
                    i = random.randrange(len(pts))
                    theta = random.random() * 2 * math.pi
                    c = (pts[i][0] + d*math.cos(theta), pts[i][1] + d*math.sin(theta))
                    if valid(c, pts, d, r):
                        pts.append(c)
                        constructed = True
                        break

            else:  # 'none'
                # строим точку далеко от всех (> r)
                for _ in range(1000):
                    theta = random.random() * 2 * math.pi
                    R = random.uniform(r + d, r_max)
                    c = (R*math.cos(theta), R*math.sin(theta))
                    if all(dist(c, p) > r for p in pts):
                        pts.append(c)
                        constructed = True
                        break

        if not constructed:
            raise RuntimeError(
                "Не удалось разместить новую точку — попробуйте увеличить r_max или уменьшить N."
            )

    return pts


def plot_and_save_points(pts, d, r, filename='points.png', dpi=300):
    """
    Визуализация: точки + «короткие» отрезки (dist≈d),
    сохранение результата в файл.
    """
    plt.figure(figsize=(6,6))
    xs, ys = zip(*pts)
    plt.scatter(xs, ys, color='k', s=50, zorder=2)

    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if abs(dist(pts[i], pts[j]) - d) < 1e-6:
                x0, y0 = pts[i]
                x1, y1 = pts[j]
                plt.plot([x0, x1], [y0, y1], 'r-', lw=2, zorder=1)
    k = r / d
    plt.axis('equal')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    #plt.show()
    plt.close()
    print(f"Изображение сохранено в '{filename}'")


if __name__ == "__main__":
    N = 29
    d = 1.0
    r = 1.0
    r_max = 10.0
    seed = 1

    pts = build_points(N=N, d=d, r=r, r_max=r_max, seed=seed)
    plot_and_save_points(pts, d, r, filename='points1.0.png')