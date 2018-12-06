import numpy as np
import math
from slam.utils import normalize_angle


class EKFSLAMKnown:
    def __init__(self, pose, num_landmarks, motion_command):
        self.rsize = 3
        self.lmsize = 2

        landmark_pose = np.zeros((2 * num_landmarks, 1))

        self.mean = np.concatenate((np.copy(pose), landmark_pose))
        self.cov = np.zeros((self.mean.size, self.mean.size))
        lm_cov = np.eye(2 * num_landmarks)
        np.fill_diagonal(
            lm_cov,
            10 ** 10
        )
        self.cov[self.rsize:, self.rsize:] = lm_cov

        self.motion_command = motion_command

    def get_mu_lid(self, lid):
        return self.rsize + self.lmsize * (lid - 1)

    def get_landmark(self, lid):
        mu_lid = self.get_mu_lid(lid)

        lx = self.mean[mu_lid, :].item(0)
        ly = self.mean[mu_lid + 1, :].item(0)

        return (lx, ly)

    def set_landmark(self, lid, lx, ly):
        mu_lid = self.get_mu_lid(lid)

        self.mean[mu_lid, :] = lx
        self.mean[mu_lid + 1, :] = ly

    def predict(self, command):
        robot_pose = self.mean[:self.rsize, :]
        self.mean[:self.rsize, :] = self.motion_command(
            robot_pose,
            command
        )

        # TODO: Put in motion model
        theta_n = normalize_angle(self.mean.item(2))
        rot1, trans, rot2 = command

        ang = normalize_angle(theta_n + rot1)
        Gtx = np.matrix([
            [1, 0, - trans * math.sin(ang)],
            [0, 1, trans * math.cos(ang)],
            [0, 0, 1],
        ])

        lmsize = self.mean.shape[0] - self.rsize

        r1zeros = np.zeros((self.rsize, lmsize))
        r2zeros = np.copy(r1zeros.T)

        gr1 = np.concatenate((Gtx, r1zeros), axis=1)
        gr2 = np.concatenate((r2zeros, np.eye(lmsize)), axis=1)
        Gt = np.concatenate((gr1, gr2))

        # motion noise
        Rtx = np.matrix([
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.01]
        ])

        rr1zeros = np.zeros((self.rsize, lmsize))
        rr2zeros = np.copy(rr1zeros.T)

        rr1 = np.concatenate(
            (Rtx, rr1zeros),
            axis=1
        )
        rr2 = np.concatenate(
            (rr2zeros, np.zeros((lmsize, lmsize))),
            axis=1
        )
        Rt = np.concatenate((rr1, rr2))

        self.cov = Gt * self.cov * Gt.T + Rt

        return self.mean, self.cov

    def correct(self, measurements, local_map):
        rx = self.mean.item(0)
        ry = self.mean.item(1)
        rtheta = normalize_angle(self.mean.item(2))

        Htfull = np.matrix([])
        Zdiff = np.matrix([])
        for reading in measurements:
            # TODO: Put in measurement model
            lid, srange, sbearing = reading
            z_measured = np.matrix([srange, normalize_angle(sbearing)]).T

            mu_lid = self.get_mu_lid(lid)

            # Expected observation
            lx = 0
            ly = 0
            if not local_map.is_added(lid):
                lx = rx + srange * math.cos(sbearing + rtheta)
                ly = ry + srange * math.sin(sbearing + rtheta)
                local_map.add((lid, lx, ly))

                self.mean[mu_lid, :] = lx
                self.mean[mu_lid + 1, :] = ly
            else:
                lx = self.mean[mu_lid, :].item(0)
                ly = self.mean[mu_lid + 1, :].item(0)

            dx = lx - rx
            dy = ly - ry

            delta = np.matrix([dx, dy]).T
            q = delta.T * delta
            z_expected = np.matrix([
                math.sqrt(q),
                normalize_angle(np.arctan2(dy, dx) - rtheta)
            ]).T
            qst = math.sqrt(q)
            # Measurement jacobian
            Htt = np.matrix([
                [-qst * dx, -qst * dy, 0, qst * dx, qst * dy],
                [dy, -dx, -q, -dy, dx]
            ])
            Htt = np.multiply((1.0 / q), Htt)

            F = np.zeros((5, self.mean.size))
            F[:self.rsize, :self.rsize] = np.eye(self.rsize)
            F[self.rsize:, mu_lid:mu_lid + self.lmsize] = np.eye(self.lmsize)

            Ht = Htt * F

            Htfull = np.concatenate((Htfull, Ht)) if Htfull.size else np.copy(Ht)

            diff = z_measured - z_expected
            # Important to normalize_angles
            diff[1] = normalize_angle(diff.item(1))

            Zdiff = np.concatenate((Zdiff, diff)) if Zdiff.size else np.copy(diff)

        # measurement noise
        Qt = np.eye(Zdiff.shape[0]) * 0.01

        Kgain = self.cov * Htfull.T * np.linalg.inv(Htfull * self.cov * Htfull.T + Qt)

        self.mean = self.mean + Kgain * Zdiff
        self.cov = (np.eye(self.mean.size) - Kgain * Htfull) * self.cov

        return self.mean, self.cov
