import math
import random
import numpy as np


def normalize_angle(phi):
    return phi - 2*math.pi * math.floor((phi + math.pi) / (2*math.pi))


# Zero mean
def sample_normal_distribution(b):
    return np.random.normal(0, b, 1)[0]


def sample_triangular_distribution(b):
    return b * random.uniform(-1, 1) * random.uniform(-1, 1)
