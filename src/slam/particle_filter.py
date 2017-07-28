import math
import numpy as np
from slam.utils import normalize_angle


class ParticleFilter:
    def __init__(self, init_pose, num_particles, motion_model, sampler,
    resampler):
        if not motion_model or not sampler:
            raise RuntimeError('Provide motion_model and sampler')
        if not resampler:
            raise RuntimeErro('Provide resampling algorithm')
        if num_particles < 1:
            raise RuntimeError('Number of particles must be positive')

        self.particles = []
        self.motion_model = motion_model
        self.sampler = sampler
        self.resampler = resampler

        self._init(init_pose, num_particles)

    def _init(self, init_pose, num_particles):
        weight = 1.0 / num_particles
        for i in range(num_particles):
            particle = (weight, np.copy(init_pose))
            self.particles.append(particle)

    def predict(self, command, noise):
        for i, particle in enumerate(self.particles):
            weight, pose = particle
            pose = self.motion_model(pose, command, noise, self.sampler)
            self.particles[i] = (weight, np.copy(pose))

    def correct(self, measurements, sensor_noise, landmark_map):
        normalizer = 0
        for i, particle in enumerate(self.particles):
            weight, pose = particle
            rx = pose.item(0)
            ry = pose.item(1)
            rtheta = pose.item(2)

            vrange, vbearing = sensor_noise

            # Matrix of measurement differences
            Zdiff = np.matrix([])
            for reading in measurements:
                lid, srange, sbearing = reading
                # Sensor measurement
                z_measured = np.matrix([srange, sbearing]).T

                lx, ly = landmark_map.get(lid)

                dx = lx - rx
                dy = ly - ry

                delta = np.matrix([dx, dy]).T
                q = delta.T * delta
                # Expected (predicted) measurement
                z_expected = np.matrix([
                    math.sqrt(q),
                    normalize_angle(np.arctan2(dy, dx) - rtheta)
                ]).T

                # Difference between measured and expected
                diff = z_expected - z_measured
                diff[1] = normalize_angle(diff.item(1))
                # Collect all measurement differences
                Zdiff = np.concatenate((Zdiff, diff)) if Zdiff.size else np.copy(diff)

            # Making sensor noise matrix with different diagonal elements
            Qdiag = np.array(
                [[vrange, vbearing] for i in range(len(measurements))]
            )
            # Flatten the array
            Qdiag.shape = (len(sensor_noise) * len(measurements), )
            Qt = np.diag(Qdiag)
            # Qt = np.eye(Zdiff.shape[0]) * 0.1

            # Normal distribution is probably a good idea 
            # Highest weight when (z_expected - z_measured) is 0
            denom = 1 / math.sqrt( np.linalg.det(2*math.pi*Qt) )
            new_weight = denom * math.exp(-1/2 * Zdiff.T * np.linalg.pinv(Qt) * Zdiff)

            normalizer = normalizer + new_weight
            self.particles[i] = (new_weight, np.copy(pose))

        self.particles = [(weight/normalizer, pose) for weight, pose in self.particles]

    def resample(self):
        self.particles = self.resampler(self.particles)


def low_variance_resampling(particles):
    new_particles = []

    Jinv = 1 / len(particles)
    r = np.random.uniform(0, Jinv)
    # weight
    c = particles[0][0]
    i = 0
    for j in range(0, len(particles)):
        # Or j-1 if out of range
        U = r + (j) * Jinv
        while U > c:
            i = i + 1
            c = c + particles[i][0]
        new_particles.append(particles[i])

    return new_particles



