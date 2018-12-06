import numpy as np
import math
from slam.utils import normalize_angle


class EKFLocalizationKnown:
    def __init__(self, pose, motion_command=None):
        # Main estimate
        self.mean = np.copy(pose)
        self.cov = np.zeros((self.mean.size, self.mean.size))

        # Intermediate variables
        self._mean = np.copy(self.mean)
        self._cov = np.copy(self.cov)

        self.motion_command = motion_command

    def predict(self, command):
        self._mean = self.motion_command(self.mean, command)

        # TODO: Put in motion model
        theta_n = normalize_angle(self.mean.item(2))
        rot1, trans, rot2 = command

        ang = normalize_angle(theta_n + rot1)
        Gt = np.matrix([
            [1, 0, - trans * math.sin(ang)],
            [0, 1, trans * math.cos(ang)],
            [0, 0, 1],

        ])

        # motion noise
        Rt = np.matrix([
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.01]
        ])

        self._cov = Gt * self.cov * Gt.T + Rt

        return self._mean, self._cov

    def correct(self, measurements, local_map):
        # measurement noise
        Qt = np.eye(self._mean.size - 1) * 0.01

        rx = self._mean.item(0)
        ry = self._mean.item(1)
        rtheta = normalize_angle(self._mean.item(2))

        # TODO: Fix kalman gain calculation
        for reading in measurements:
            # TODO: Put in measurement model
            lid, srange, sbearing = reading
            z_measured = np.matrix([srange, normalize_angle(sbearing)]).T

            # Expected observation
            lx, ly = local_map.get(lid)
            dx = lx - rx
            dy = ly - ry
            delta = np.matrix([dx, dy]).T
            q = delta.T * delta
            z_expected = np.matrix([
                math.sqrt(q),
                normalize_angle(np.arctan2(dy, dx) - rtheta)
            ]).T
            # Jacobian
            Ht = np.matrix([
                [-math.sqrt(q) * dx, -math.sqrt(q) * dy, 0],
                [dy, -dx, -q]
            ])
            Ht = np.multiply((1.0 / q), Ht)

            Kgain = self._cov * Ht.T * np.linalg.inv(Ht * self._cov * Ht.T + Qt)

            diff = z_measured - z_expected
            diff[1] = normalize_angle(diff.item(1))

            self._mean = self._mean + Kgain * diff
            self._cov = (np.eye(self._mean.size) - Kgain * Ht) * self._cov

        self.mean = np.copy(self._mean)
        self.cov = np.copy(self._cov)

        return self.mean, self.cov
