import math
import random
import numpy as np


def normalize_angle(phi):
    return phi - 2*math.pi * math.floor((phi + math.pi) / (2*math.pi))


def sample_normal_distribution(b):
    return np.random.normal(0, b, 1)[0]


def sample_triangular_distribution(b):
    return b * random.uniform(-1, 1) * random.uniform(-1, 1)


def log2prob(l):
    return 1 - (1 / (1 + np.exp(l)))


def prob2log(p):
    return np.log(p / (1 - p))


def vector2transform2D(vector):
    angle = vector.item(2)
    cs = math.cos(angle)
    sn = math.sin(angle)
    return np.matrix([
        [cs, -sn, vector.item(0)],
        [sn, cs, vector.item(1)],
        [0, 0, 1]
    ])


def transform2vector2D(t):
    return np.matrix([
        [t[0, 2]],
        [t[1, 2]],
        [np.arctan2(t[1, 0], t[0, 0])]
    ])


def bresenham_line(start, end):
    # Copy/paste from roguebasin
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


