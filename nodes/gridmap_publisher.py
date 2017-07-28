#!/usr/bin/env python
import numpy as np
import math
import sys
from slam.gridmap import GridMap, LaserDataMatlab

import rospy
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData
from rospy.numpy_msg import numpy_msg


def timestep_gen(timesteps):
    for t in timesteps:
        yield t


def gridmap_publisher():
    gridmap_pub = rospy.Publisher('gridmap', OccupancyGrid, queue_size=10)
    rospy.init_node('gridmap_publisher', sys.argv)
    rospy.loginfo('Initialized gridmap_publisher')

    # Gridmap 
    laser_data = LaserDataMatlab(sys.argv[1])
    gridmap = GridMap(grid_size=0.3)
    gridmap.init_from_laserdata(laser_data)

    # Setup ros message
    timesteps = timestep_gen(laser_data.get_timestep_list())

    map_meta = MapMetaData()
    # map_meta.map_load_time = rospy.Time().now()
    map_meta.resolution = gridmap._grid_size

    grid_map = gridmap.get_prob_map().T * 100
    grid_map = grid_map.astype(np.int8)

    # Width = cols
    map_meta.width = grid_map.shape[1]
    # Height = rows
    map_meta.height = grid_map.shape[0]

    map_meta.origin.position.x = 0
    map_meta.origin.position.y = 0
    map_meta.origin.position.z = 0

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            t = timesteps.next()
            robot_pose = laser_data.get_pose(t)
            range_scan = laser_data.get_range_scan(t)
            gridmap.update(robot_pose, range_scan)
        except StopIteration:
            pass

        grid_map = gridmap.get_prob_map().T * 100
        grid_map = grid_map.astype(np.int8)

        h = Header()
        h.stamp = rospy.Time.now()
        map_meta.map_load_time = rospy.Time().now()
        gridmap_pub.publish(
            header=h,
            info=map_meta,
            data=grid_map.flatten()
        )

        rate.sleep()


if __name__ == '__main__':
    try:
        gridmap_publisher()
    except rospy.ROSInterruptException:
        pass
