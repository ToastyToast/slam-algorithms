import numpy as np
import math
from slam.utils import normalize_angle


def odometry_command(pose, command):
    rot1, trans, rot2 = command
    theta_old = normalize_angle(pose.item(2))

    normalized = normalize_angle(theta_old + rot1)
    update_vec = np.matrix([
        trans * math.cos(normalized),
        trans * math.sin(normalized),
        normalize_angle(rot1 + rot2)
    ]).T

    pose = pose + update_vec
    pose[2] = normalize_angle(pose.item(2))

    return np.copy(pose)


def odometry_sample(pose, command, noise, sample=None):
    if not sample:
        raise ValueError("Provide a sampler")

    rot1, trans, rot2 = command
    r1_noise, t_noise, r2_noise = noise
    theta_old = normalize_angle(pose.item(2))

    rot1_h = rot1 - sample(r1_noise)
    trans_h = trans - sample(t_noise)
    rot2_h = rot2 - sample(r2_noise)

    normalized = normalize_angle(theta_old + rot1_h)
    update_vec = np.matrix([
        trans_h * math.cos(normalized),
        trans_h * math.sin(normalized),
        normalize_angle(rot1_h + rot2_h)
    ]).T

    temp = pose + update_vec
    temp[2] = normalize_angle(temp.item(2))

    return np.copy(temp)

