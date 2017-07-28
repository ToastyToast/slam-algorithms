import numpy as np


class BaseRobot:
    def __init__(self, x, y, theta, motion_command):
        self._pose = np.matrix([x, y, theta]).T
        self._motion_command = motion_command

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, new_pose):
        self._pose = np.copy(new_pose)

    @pose.deleter
    def pose(self):
        self._pose = None
        del self._pose

    @property
    def motion(self):
        return self._motion_command

    @motion.setter
    def motion(self, motion_model):
        self._motion_command = motion_model

    @motion.deleter
    def motion(self):
        del self._motion_command

    def motion_command(self, command):
        self._pose = self._motion_command(self._pose, command)
