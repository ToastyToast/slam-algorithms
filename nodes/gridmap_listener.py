#!/usr/bin/env python
import math
import sys
import time
import threading

import numpy as np
from slam.gridmap import GridMap
from slam.utils import normalize_angle

import rospy
import tf, tf2_ros
import std_msgs.msg, geometry_msgs.msg, nav_msgs.msg, sensor_msgs.msg


class GridmapListener:
    def __init__(self):
        self.lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.scan_sub = rospy.Subscriber('scan', sensor_msgs.msg.LaserScan, self._scan_cb)

        self.map_pub = rospy.Publisher('gridmap', nav_msgs.msg.OccupancyGrid, queue_size=1)

        self.last_scan_msg = None
        self.last_tf_msg = None
        self.gridmap = GridMap(grid_size=0.1)
        self.gridmap.init_gridmap((50, 50))

        self.gridmap_msg = nav_msgs.msg.OccupancyGrid()
        self.gridmap_msg.info.map_load_time = rospy.Time.now()
        self.gridmap_msg.info.resolution = self.gridmap._grid_size
        self.gridmap_msg.info.width = self.gridmap._map_size[1]
        self.gridmap_msg.info.height = self.gridmap._map_size[0]


    def _scan_cb(self, msg):
        with self.lock:
            self.last_scan_msg = msg

            trans = None
            try:
                trans = self.tf_buffer.lookup_transform('base_footprint',
                    'odom', rospy.Time.now())
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException
            ):
                pass

            if trans:
                self.last_tf_msg = trans

    def update_map(self):
        with self.lock:
            if not self.last_scan_msg or not self.last_tf_msg:
                return

            rx = self.last_tf_msg.transform.translation.x
            ry = self.last_tf_msg.transform.translation.y

            quat = (
                self.last_tf_msg.transform.rotation.x,
                self.last_tf_msg.transform.rotation.y,
                self.last_tf_msg.transform.rotation.z,
                self.last_tf_msg.transform.rotation.w,
            )
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                quat
            )

            rtheta = normalize_angle(yaw)
            rob_pose = np.matrix([rx, ry, rtheta]).T

            # TODO: Calculate laser offset
            range_scan = {
                'ranges': np.array(self.last_scan_msg.ranges, dtype=np.float32),
                'maximum_range': self.last_scan_msg.range_max,
                'start_angle': self.last_scan_msg.angle_min,
                'angular_resolution': self.last_scan_msg.angle_increment,
                'laser_offset': np.matrix([0, 0, 0])
            }
            s_angle = range_scan['start_angle']
            ang_res = range_scan['angular_resolution']
            num_beams = len(range_scan['ranges'])
            print(s_angle, ang_res, s_angle + ang_res * num_beams)
            print(
                self.last_scan_msg.angle_min,
                self.last_scan_msg.angle_increment,
                self.last_scan_msg.angle_max,
            )
            rospy.loginfo('Updating map')

            self.gridmap.update(rob_pose, range_scan)

            self.gridmap_msg.header.stamp = rospy.Time.now()
            prob_map = self.gridmap.get_prob_map() * 100
            self.gridmap_msg.data = prob_map.flatten()
            self.map_pub.publish(self.gridmap_msg)



def gridmap_listener():
    rospy.init_node('gridmap_listener')
    rospy.loginfo('Initialized gridmap_listener')

    listener = GridmapListener()

    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():

        listener.update_map()

        rate.sleep()



if __name__ == '__main__':
    try:
        gridmap_listener()
    except rospy.ROSInterruptException:
        pass
