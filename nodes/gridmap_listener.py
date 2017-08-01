#!/usr/bin/env python
import math
import sys
import time
import threading

import numpy as np
from slam.gridmap import GridMap

import rospy
import tf, tf2_ros
import std_msgs.msg, geometry_msgs.msg, nav_msgs.msg, sensor_msgs.msg


def gridmap_listener():
    rospy.init_node('gridmap_listener')
    rospy.loginfo('Initialized gridmap_listener')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        try:
            trans = tf_buffer.lookup_transform('base_footprint', 'odom', rospy.Time())
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):
            rate.sleep()
            continue

        rate.sleep()



if __name__ == '__main__':
    try:
        gridmap_listener()
    except rospy.ROSInterruptException:
        pass
