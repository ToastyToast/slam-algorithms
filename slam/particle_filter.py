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

    def correct(self, measurements):
        pass

    def resample(self):
        self.particles = self.resampler(self.particles)


def low_variance_resampling(particles):
    new_particles = []

    Jinv = 1 / len(particles)
    r = np.random.normal(0, Jinv)
    # weight
    c = particles[0][0]
    i = 1
    for j in range(0, len(particles)):
        U = r + (j - 1) * Jinv
        while U > c:
            i = i + 1
            c = c + particles[i][0]
        new_particles.append(particles[i])

    return new_particles



