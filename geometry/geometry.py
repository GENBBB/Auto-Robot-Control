"""This module provides functionality for working with polygons"""
import math
import numpy as np
from skspatial.objects import Point


def pseudo_dot(a: Point, b: Point) -> float:
    """
    Return pseudo-scalar product a and b

    Parameters
    ----------
    a, b: np.array
        2D points

    Returns
    -------
    pseudo_dot: float
        Pseudo_dot of a, b

    """
    np.dot(a, b)
    return a[0] * b[1] - a[1] * b[0]


def is_convex(points: list[Point], traversal: str = 'r') -> bool:
    """
    Checking a polygon for convexity
    It is assumed that the polygon is closed, and the first connection point is from the last
    Depending on the traversal, it is considered that the vertices of the polygon are located in the order of right
    or left traversal

    Parameters
    ----------
    points: list of scikit-spatial.objects.Point
        Points of a closed polygon
    traversal: string, either 'r', 'l', optional
        If the traversal is 'r', then it is considered that the vertices of the polygon are located in the order of
        right traversal. If 'l', then left

    Returns
    ---------
    bool:
        True is polygon is convex and false is not
    """
    if traversal not in ['r', 'l']:
        raise ValueError("sign option must be one of 'r', 'l'")
    if traversal == 'r':
        sign = 1
    else:
        sign = -1
    for a in points:
        b = next(a, None)
        if b is None:
            b = points[0]
        c = next(b, None)
        if c is None:
            c = points[0]
        if np.sign(pseudo_dot(b - a, c - b)) != sign and np.sign(pseudo_dot(b - a, c - b)) != 0:
            return False
    return True


def viewing_angels_circle(point: Point, center: Point, radius: float) -> tuple[float, float]:
    """
    Calculate from what angles relative to the x-axis the circle is visible when going around to the right
    Angles range [-pi/2, 5pi/2], but difference between them is no more than pi

    Parameters
    ----------
    point: scikit-spatial.objects.Point
        The point from which one looks at the circle
    center: scikit-spatial.objects.Point
        Circle center point
    radius: float
        Radius of circle

    Returns
    -------
    float
        min_angle
    float
        max_angle

    Raises
    -------
    ValueError
        If point belongs to the circle
    """
    v = center - point
    distance = np.linalg.norm(v)
    if distance < radius:
        raise ValueError('The point belongs to the circle')
    alpha = np.arcsin(radius / distance)
    beta = np.arccos(v[0]/distance)
    if v[1] < 0:
        beta = -beta
    left = beta - alpha
    right = beta + alpha
    if left <= 0:
        left += 2 * math.pi
        right += 2 * math.pi
    if right >= 2 * math.pi:
        left -= 2 * math.pi
        right -= 2 * math.pi
    return left, right
