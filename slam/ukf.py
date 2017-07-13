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
        self.cov = np.eye(state_size) * 0.001
        self.state_size = state_size

        # Unscented transform parameters
        n = state_size
        self.alpha = 0.5
        self.beta = 2
        self.kappa = 50
        self.lambd = self.alpha*self.alpha*(n+self.kappa)-n

    def predict(self, motion_model_transform, command, noise):
        sigma_points, w_m, w_c = compute_sigma_points(
            self.mean, self.cov, self.lambd, self.alpha, self.beta
        )
        # Normalize
        for i in range(sigma_points.shape[1]):
            sigma_points[2, i] = normalize_angle(sigma_points[2, i])

        # Angles are normalized in motion model
        sigma_points = motion_model_transform(sigma_points, command)

        # TODO: rewrite to normalize differences
        self.mean, self.cov = recover_gaussian(sigma_points, w_m, w_c)

        # fix angles
        average_theta = 0
        avg_x = avg_y = 0
        for i in range(sigma_points.shape[1]):
            avg_x = avg_x + w_m[i]*math.cos(sigma_points[2, i])
            avg_y = avg_y + w_m[i]*math.sin(sigma_points[2, i])
        average_theta = normalize_angle(np.arctan2(avg_y, avg_x))
        self.mean[2] = average_theta

        # fix angles in covariance matrix too
        for i in range(self.cov.shape[1]):
            self.cov[2, i] = normalize_angle(self.cov[2, i])

        Rt = np.diag(noise)
        self.cov = self.cov + Rt

        return self.mean, self.cov

    def correct(self, meas_model_transform, measurements, noise, landmark_map):
        sigma_points, w_m, w_c = compute_sigma_points(
            self.mean, self.cov, self.lambd, self.alpha, self.beta
        )
        # Normalize
        for i in range(sigma_points.shape[1]):
            sigma_points[2, i] = normalize_angle(sigma_points[2, i])

        for reading in measurements:
            # Real measurement
            lid, srange, sbearing = reading
            z_measured = np.matrix([srange, normalize_angle(sbearing)]).T

            # Predicted measurement
            # Transform points through measurment_model
            # Resulting matrix has 2 rows (range, bearing) and same number of columns
            z_points = meas_model_transform(sigma_points, reading, landmark_map)

            # Recover gaussian for predicted measurement
            # TODO: rewrite to normalize differences
            zt, St = recover_gaussian(z_points, w_m, w_c)

            # Normalize bearing
            average_bearing = 0
            avg_x = avg_y = 0
            for i in range(z_points.shape[1]):
                # normalize 
                z_points[1, i] = normalize_angle(z_points[1, i])
                # weighted sums of cosines and sines of the angle
                avg_x = avg_x + w_m[i]*math.cos(z_points[1, i])
                avg_y = avg_y + w_m[i]*math.sin(z_points[1, i])
            average_bearing = normalize_angle(np.arctan2(avg_y, avg_x))
            zt[1] = average_bearing

            # normalize bearing in covariance matrix 
            for i in range(St.shape[1]):
                St[1, i] = normalize_angle(St[1, i])

            # Measurement noise
            Qt = np.diag(noise)
            St = St + Qt

            sigma_x_z = np.zeros((self.mean.shape[0], zt.shape[0]))
            for i in range(z_points.shape[1]):
                s_diff = sigma_points[:, [i]] - self.mean
                z_diff = z_points[:, [i]] - zt

                # Normalize angles
                s_diff[2] = normalize_angle(s_diff[2])
                z_diff[1] = normalize_angle(z_diff[1])

                sigma_x_z = sigma_x_z + w_c[i] * s_diff * z_diff.T

            # Normalization time
            for i in range(sigma_x_z.shape[1]):
                sigma_x_z[2, i] = normalize_angle(sigma_x_z[2, i])

            # broadcasting error if not matrix
            sigma_x_z = np.matrix(sigma_x_z)

            # Kalman gain
            Kt = sigma_x_z * np.linalg.pinv(St)

            # Normalize
            z_diff = z_measured - zt
            z_diff[1] = normalize_angle(z_diff[1])

            self.mean = self.mean + Kt * (z_diff)
            self.cov = self.cov - Kt * St * Kt.T

            # normalize angle
            self.mean[2] = normalize_angle(self.mean[2])
            for i in range(self.cov.shape[1]):
                self.cov[2, i] = normalize_angle(self.cov[2, i])

        return self.mean, self.cov


def odometry_model_transform(points, command):
    for i in range(points.shape[1]):
        pose_col = np.copy(points[:, [i]])
        points[:, [i]] = odometry_command(pose_col, command)
    return points


def measurement_model_transform(points, measurement, landmark_map):
    z_points = np.zeros((points.shape[0]-1, points.shape[1]))
    for i in range(points.shape[1]):
        rx = points[0, i]
        ry = points[1, i]
        rtheta = points[2, i]

        lid, srange, sbearing = measurement
        z_measured = np.matrix([srange, normalize_angle(sbearing)]).T

        lx, ly = landmark_map.get(lid)

        dx = lx - rx
        dy = ly - ry
        delta = np.matrix([dx, dy]).T
        q = delta.T * delta
        z_expected = np.matrix([
            math.sqrt(q),
            normalize_angle(np.arctan2(dy, dx) - rtheta)
        ]).T

        z_points[:, [i]] = z_expected
    return z_points


def compute_sigma_points(mu, sigma, lambd, alpha, beta):
    n = mu.shape[0]
    sigma_points = np.zeros((n, 2*n+1))

    sigma_points[:, [0]] = mu

    # mroot = np.linalg.cholesky(sigma)
    mroot = sqrtm(sigma)
    mroot = math.sqrt(n+lambd) * mroot

    # compute sigma points
    for i in range(1, n+1):
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

