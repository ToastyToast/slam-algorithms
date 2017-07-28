import math
import numpy as np
import scipy.io as sio
from slam.utils import (bresenham_line, prob2log, log2prob,
    vector2transform2D, transform2vector2D)


class LaserDataMatlab:
    def __init__(self, filename):
        self.filename = filename
        laser_content = None
        try:
            laser_content = sio.loadmat(filename)
        except FileNotFoundError:
            raise ValueError('Provide a laser scan .mat file')

        self._laser = laser_content['laser']
        self._poses = np.asmatrix(np.vstack(self._laser['pose'][0]))

    @property
    def poses(self):
        return self._poses

    def get_timestep_list(self):
        return list(range(len(self.poses)))

    def get_pose(self, timestep):
        return self._poses[timestep].T

    def get_range_scan(self, timestep):
        range_scan = self._laser[0, timestep]
        ranges = range_scan['ranges'][0]
        max_range = range_scan['maximum_range'][0][0]
        start_angle = range_scan['start_angle'][0][0]
        angular_res = range_scan['angular_resolution'][0][0]
        laser_offset = np.asmatrix(range_scan['laser_offset'])

        return {
            'ranges': ranges,
            'maximum_range': max_range,
            'start_angle': start_angle,
            'angular_resolution': angular_res,
            'laser_offset': laser_offset
        }

    def init_gridmap_from_data(self, gridmap):
        pose_x_min = np.min(self._poses[:, 0])
        pose_x_max = np.max(self._poses[:, 0])
        pose_y_min = np.min(self._poses[:, 1])
        pose_y_max = np.max(self._poses[:, 1])

        map_borders = (pose_x_min-gridmap._border, pose_x_max+gridmap._border,
                       pose_y_min-gridmap._border, pose_y_max+gridmap._border)

        offset_x = map_borders[0]
        offset_y = map_borders[2]
        gridmap._offset = (offset_x, offset_y)

        gridmap._map_size_meters = (map_borders[1]-offset_x, map_borders[3]-offset_y)
        gridmap._map_size = tuple([math.ceil(dim/gridmap._grid_size) for dim in gridmap._map_size_meters])

        log_odds_prior = prob2log(gridmap._prior)
        gridmap._grid_map = np.ones(gridmap._map_size).T * log_odds_prior


class GridMap:
    def __init__(self, grid_size=0.5, border=30):
        # Default 
        self._prior = 0.5
        self._prob_occ = 0.9
        self._prob_free = 0.35

        self._border = border
        self._grid_size = grid_size

        self._offset = None
        self._map_size_meters = None
        self._map_size = None
        self._grid_map = None

    def init_from_laserdata(self, laserdata):
        laserdata.init_gridmap_from_data(self)

    def inv_sensor_model(self, scan, pose):
        map_update = np.zeros(self._grid_map.shape)

        rob_trans = vector2transform2D(pose)
        robot_pose_map_frame = GridMap.world_to_map_coordinates(
            pose[0:2, :], self._grid_size, self._offset
        )

        laser_end_points = GridMap.laser_as_cartesian(scan, 30)
        laser_end_points = rob_trans * laser_end_points

        laser_end_map_frame = GridMap.world_to_map_coordinates(
            laser_end_points[0:2, :], self._grid_size, self._offset
        )

        for col in range(laser_end_map_frame.shape[1]):
            rx = int(robot_pose_map_frame.item(0))
            ry = int(robot_pose_map_frame.item(1))
            lx = int(laser_end_map_frame.item((0, col)))
            ly = int(laser_end_map_frame.item((1, col)))

            bres_points = bresenham_line((rx, ry), (lx, ly))
            for point in bres_points:
                px, py = point
                map_update[py, px] = self._grid_map[py, px] + \
                    prob2log(self._prob_free)

            map_update[py, px] = self._grid_map[py, px] + \
                prob2log(self._prob_occ)

        return map_update, robot_pose_map_frame, laser_end_map_frame

    def update(self, robot_pose, laser_scan):
        map_update, pose_map_frame, laser_map_frame = self.inv_sensor_model(
            laser_scan, robot_pose
        )

        log_odds_prior = prob2log(self._prior)
        map_update = map_update - log_odds_prior * np.ones(map_update.shape)
        self._grid_map = self._grid_map + map_update

    def get_prob_map(self):
        return np.ones(self._grid_map.shape) - log2prob(self._grid_map)

    @staticmethod
    def laser_as_cartesian(rl, max_range=15):
        ranges = rl['ranges']
        num_beams = len(ranges)
        max_range = min(max_range, rl['maximum_range'])
        idx = (ranges < max_range) & (ranges > 0)

        s_angle = rl['start_angle']
        a_res = rl['angular_resolution']
        angles = np.linspace(s_angle, s_angle+num_beams*a_res, num_beams)[idx]
        ranges = ranges[idx]

        cs = np.cos(angles)
        sn = np.sin(angles)
        points = np.vstack([
            ranges * cs,
            ranges * sn,
            np.ones( (1, len(angles)) )
        ])
        transf = vector2transform2D(rl['laser_offset'])

        return transf * points

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

