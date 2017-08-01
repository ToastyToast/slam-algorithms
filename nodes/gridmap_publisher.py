#!/usr/bin/env python
import numpy as np
import math
import sys
from slam.gridmap import GridMap, LaserDataMatlab

import rospy
import tf, tf2_ros
import geometry_msgs.msg
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from rospy.numpy_msg import numpy_msg


def timestep_gen(timesteps):
    for t in timesteps:
        yield t


def gridmap_publisher():
    gridmap_pub = rospy.Publisher('gridmap', OccupancyGrid, queue_size=1)
    pose_pub = rospy.Publisher('pose', geometry_msgs.msg.PoseStamped, queue_size=10)
    scan_pub = rospy.Publisher('scan', LaserScan, queue_size=10)
    rospy.init_node('gridmap_publisher', sys.argv)
    rospy.loginfo('Initialized gridmap_publisher')

    # Gridmap 
    laser_data = LaserDataMatlab(sys.argv[1])
    gridmap = GridMap(grid_size=0.1)
    gridmap.init_from_laserdata(laser_data)

    # Setup ros message
    timestep_list = laser_data.get_timestep_list()
    timesteps = timestep_gen(timestep_list)
    timestep_len = len(timestep_list)

    map_meta = MapMetaData()
    # map_meta.map_load_time = rospy.Time().now()
    map_meta.resolution = gridmap._grid_size

    grid_map = gridmap.get_prob_map() * 100
    grid_map = grid_map.astype(np.int8)

    # Width = cols
    map_meta.width = grid_map.shape[1]
    # Height = rows
    map_meta.height = grid_map.shape[0]

    map_meta.origin.position.x = 0
    map_meta.origin.position.y = 0
    map_meta.origin.position.z = 0

    # Broadcast robot transform

    pose_tf_br = tf2_ros.TransformBroadcaster()

    rbase_tf = geometry_msgs.msg.TransformStamped()
    rbase_tf.header.stamp = rospy.Time.now()
    rbase_tf.header.frame_id = 'map'
    rbase_tf.child_frame_id = 'robot_base'

    rate = rospy.Rate(10)

    rx = 0 - gridmap._offset[0]
    ry = 0 - gridmap._offset[1]
    rtheta = 0

    pose_msg = geometry_msgs.msg.PoseStamped()
    pose_msg.header.frame_id = 'map'

    scan_msg = LaserScan()
    scan_msg.header.frame_id = 'robot_base'

    # Get info about laser scan
    laser_scan_temp = laser_data.get_range_scan(0)

    # This info doesn't change
    start_angle = laser_scan_temp['start_angle']
    angular_resolution = laser_scan_temp['angular_resolution']

    scan_msg.angle_min = start_angle
    scan_msg.angle_increment = angular_resolution
    scan_msg.range_min = 0
    scan_msg.time_increment = 0

    robot_pose = None
    range_scan = None
    while not rospy.is_shutdown():
        try:
            t = next(timesteps)
            robot_pose = laser_data.get_pose(t)
            range_scan = laser_data.get_range_scan(t)
            gridmap.update(robot_pose, range_scan)
        except StopIteration:
            robot_pose = laser_data.get_pose(timestep_len-1)
            range_scan = laser_data.get_range_scan(timestep_len-1)

        grid_map = gridmap.get_prob_map() * 100
        grid_map = grid_map.astype(np.int8)

        rx = robot_pose.item(0) - gridmap._offset[0]
        ry = robot_pose.item(1) - gridmap._offset[1]
        rtheta = robot_pose.item(2)

        # Robot_base transform 
        rbase_tf.transform.translation.x = rx
        rbase_tf.transform.translation.y = ry
        rbase_tf.transform.translation.z = 0

        quat = tf.transformations.quaternion_from_euler(0, 0, rtheta)
        rbase_tf.transform.rotation.x = quat[0]
        rbase_tf.transform.rotation.y = quat[1]
        rbase_tf.transform.rotation.z = quat[2]
        rbase_tf.transform.rotation.w = quat[3]

        # Robot pose 
        pose_msg.pose.position.x = rx
        pose_msg.pose.position.y = ry
        pose_msg.pose.position.z = 0

        pose_quat = tf.transformations.quaternion_from_euler(0, 0, rtheta)
        pose_msg.pose.orientation.x = pose_quat[0]
        pose_msg.pose.orientation.y = pose_quat[1]
        pose_msg.pose.orientation.z = pose_quat[2]
        pose_msg.pose.orientation.w = pose_quat[3]

        # Laser scan
        # Get changing info from current scan
        num_beams = len(range_scan['ranges'])
        max_range = range_scan['maximum_range']
        laser_ranges = range_scan['ranges']

        valid_endpoints = (laser_ranges < max_range) & (laser_ranges > 0)
        laser_ranges = laser_ranges[valid_endpoints]

        scan_msg.angle_max = start_angle + num_beams * angular_resolution
        scan_msg.range_max = max_range
        scan_msg.ranges = laser_ranges
        scan_msg.time_increment = (1/50) / num_beams
        scan_msg.scan_time = rospy.Time.now().nsecs - scan_msg.scan_time

        # Publish everything
        rbase_tf.header.stamp = rospy.Time.now()
        pose_msg.header.stamp = rospy.Time.now()
        scan_msg.header.stamp = rospy.Time.now()

        pose_tf_br.sendTransform(rbase_tf)
        pose_pub.publish(pose_msg)
        scan_pub.publish(scan_msg)

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
