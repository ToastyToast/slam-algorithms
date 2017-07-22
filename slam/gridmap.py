import math
import numpy as np
import scipy.io as sio
from slam.utils import bresenham_line


class GridMap:
    def __init__(self, laser_data='', grid_size=0.5, border=30):
        # Default 
        self._prior = 0.5
        self._prob_coc = 0.9
        self._prob_free = 0.35

        self._border = border
        self._grid_size = grid_size

        self._offset = None
        self._map_size_meters = None
        self._map_size = None
        self._grid_map = None
        self._poses = None

        if laser_data:
            self.init_laser_from_mat(laser_data)

    def init_laser_from_mat(self, laser_filename):
        laser_content = None
        try:
            laser_content = sio.loadmat(laser_filename)
        except FileNotFoundError:
            raise ValueError('Provide a laser scan .mat file')

        laser = laser_content['laser']
        poses = laser['pose'][0]
        # convert to (N, M) matrix instead of (N, )
        self._poses = np.matrix([arr[0].tolist() for arr in poses])

        pose_x_min = np.min(self._poses[:, 0])
        pose_x_max = np.max(self._poses[:, 0])
        pose_y_min = np.min(self._poses[:, 1])
        pose_y_max = np.max(self._poses[:, 1])

        map_borders = (pose_x_min-self._border, pose_x_max+self._border,
                       pose_y_min-self._border, pose_y_max+self._border)

        offset_x = map_borders[0]
        offset_y = map_borders[2]
        self._offset = (offset_x, offset_y)

        self._map_size_meters = (map_borders[1]-offset_x, map_borders[3]-offset_y)
        self._map_size = tuple([math.ceil(dim/self._grid_size) for dim in self._map_size_meters])

        log_odds_prior = GridMap.prob2log(self._prior)
        self._grid_map = np.ones(self._map_size).T * log_odds_prior

    @staticmethod
    def log2prob(l):
        return 1 - (1 / (1 + np.exp(l)))

    @staticmethod
    def prob2log(p):
        return np.log(p / (1 - p))

    @staticmethod
    def world_to_map_coordinates(world_points, grid_size, offset):
        ofx, ofy = offset
        map_points = np.zeros(world_points.shape)

        for i in range(map_points.shape[1]):
            col = world_points[:, [i]]
            map_points[:, [i]] = np.array([
                [math.ceil((col[0] - ofx) / grid_size)],
                [math.ceil((col[1] - ofy) / grid_size)]
            ])

        return map_points

