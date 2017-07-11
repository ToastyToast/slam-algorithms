import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from slam.utils import normalize_angle
from slam.motion_models.odometry_model import odometry_command


class UKF:
    def __init__(self, pose):
        state_size = pose.shape[0]
        self.mean = np.zeros((state_size, 1))
        self.cov = np.eye(state_size) * 0.1
        self.state_size = state_size

        # Unscented transform parameters
        n = state_size
        self.alpha = 0.9
        self.beta = 2
        self.kappa = 1
        self.lambd = self.alpha*self.alpha*(n+self.kappa)-n
        self.lambd = 1

    def predict(self, motion_model_transform, command, noise):
        sigma_points, w_m, w_c = compute_sigma_points(
            self.mean, self.cov, self.lambd, self.alpha, self.beta
        )

        sigma_points = motion_model_transform(sigma_points, command)

        self.mean, self.cov = recover_gaussian(sigma_points, w_m, w_c)

        # fix angles
        average_theta = 0
        avg_x = avg_y = 0
        for i in range(sigma_points.shape[1]):
            avg_x = avg_x + w_m[i]*math.cos(sigma_points[2, i])
            avg_y = avg_y + w_m[i]*math.sin(sigma_points[2, i])
        average_theta = normalize_angle(np.arctan2(avg_y, avg_x))

        self.mean[2] = average_theta

        for i in range(self.cov.shape[1]):
            self.cov[2, i] = normalize_angle(self.cov[2, i])

        Rt = np.diag(noise)
        self.cov = self.cov + Rt

        return self.mean, self.cov

    def correct(self, measurements, noise, landmark_map):
        pass


def odometry_model_transform(points, command):
    for i in range(points.shape[1]):
        pose_col = np.copy(points[:, [i]])
        points[:, [i]] = odometry_command(pose_col, command)
    return points


def compute_sigma_points(mu, sigma, lambd, alpha, beta):
    n = mu.shape[0]
    sigma_points = np.zeros((n, 2*n+1))

    sigma_points[:, [0]] = mu


    # mroot = np.linalg.cholesky(sigma)
    mroot = sqrtm(sigma)
    mroot = math.sqrt(n+lambd) * mroot

    # compute sigma points
    for i in range(1, n+1):
        # nice indexing
        sigma_points[:,[i]] = mu + mroot[:, [i-1]]

    for i in range(n+1, 2*n+1):
        sigma_points[:,[i]] = mu - mroot[:, [i-n-1]]

    # compute weights for mean and covariance recovery
    w_m = np.zeros(2*n+1)
    w_c = np.zeros(2*n+1)

    w_m[0] = lambd / (n + lambd)
    w_c[0] = w_m[0] + (1 - alpha**2 + beta)
    weight = 1 / ( 2 * (n + lambd))
    for t in range(1,2*n+1):
        w_m[t] = weight
        w_c[t] = weight

    return sigma_points, w_m, w_c


def recover_gaussian(sigma_points, w_m, w_c):
    n = sigma_points.shape[0]
    n_col = sigma_points.shape[1]

    mu = np.zeros((n, 1))

    # Recover mean
    for i in range(0, n_col):
        mu = mu + w_m[i] * sigma_points[:, [i]]

    # Recover covariance
    sigma = np.zeros((n, n))
    for i in range(0, n_col):
        sigma = sigma + w_c[i]*(sigma_points[:, [i]] - mu) * (sigma_points[:, [i]] - mu).T

    return mu, sigma

