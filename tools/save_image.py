#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def callback(msg):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite("/home/ig-handle/.gemini/antigravity/brain/1cd36689-c3d5-4357-b469-ba0e0d4555e4/camera_debug.png", cv_image)
        rospy.loginfo("Image saved to camera_debug.png")
        rospy.signal_shutdown("Image saved")
    except Exception as e:
        rospy.logerr(e)

rospy.init_node('image_saver')
rospy.Subscriber("/camera/rgb/image_raw", Image, callback)
rospy.spin()
