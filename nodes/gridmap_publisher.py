#!/usr/bin/env python

import numpy as np

import rospy
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData
from rospy.numpy_msg import numpy_msg


def gridmap_publisher():
    gridmap_pub = rospy.Publisher('gridmap', OccupancyGrid, queue_size=10)
    rospy.init_node('gridmap_publisher')
    rospy.loginfo('Initialized gridmap_publisher')

    map_meta = MapMetaData()
    map_meta.map_load_time = rospy.Time().now()
    map_meta.resolution = 0.5

    grid_map = np.random.randint(100, size=(200, 200), dtype=np.uint8)

    map_meta.width = grid_map.shape[0]
    map_meta.height = grid_map.shape[1]

    map_meta.origin.position.x = 0
    map_meta.origin.position.y = 0
    map_meta.origin.position.z = 0

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        h = Header()
        h.stamp = rospy.Time.now()
        map_meta.map_load_time = rospy.Time().now()
        gridmap_pub.publish(
            header=h,
            info=map_meta,
            data=np.random.randint(
                100,
                size=(200, 200),
                dtype=np.uint8
            ).flatten()
        )

        rospy.loginfo('Publishing map')
        rate.sleep()


if __name__ == '__main__':
    try:
        gridmap_publisher()
    except rospy.ROSInterruptException:
        pass
