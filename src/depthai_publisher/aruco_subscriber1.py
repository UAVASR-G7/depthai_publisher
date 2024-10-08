#!/usr/bin/env python3

import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32
from geometry_msgs.msg import Point, PoseStamped
from spar_msgs.msg import ArucoLocalisation
from math import *
import numpy as np
import threading 
import time

class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'

    def __init__(self):
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)

        # Use for emulator
        # self.sub_pose = rospy.Subscriber("uavasr/pose", PoseStamped, self.callback_pose)
        self.sub_pose = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.callback_pose)

        self.aruco_pub_inf = rospy.Publisher('/processed_aruco/localisation', ArucoLocalisation, queue_size=10)
        self.br = CvBridge()
        self.last_msg_time = rospy.Time(0)
        self.lock = threading.Lock()

        # Camera Variables
        self.camera_FOV_x = 54 * (pi / 180) # [rad]
        self.camera_FOV_y = 66 * (pi / 180) # [rad]

        # Pose
        self.current_location = Point()

        self.pub_aruco_vocal = rospy.Publisher('vocal/aruco', Int32, queue_size=10)

        # Aruco Variables
        self.desired_aruco_id = 6 # Need to change to desired Aruco
        self.previous_aruco_id = -1
        self.FoundAruco = False
        self.aruco_detected = False
        self.detected_positions = []
        self.last_seen_time = time.time()
        self.detection_timeout = 1.0  # Time in seconds to wait after losing detection to publish average position
        self.detected_markers = {}  # Dictionary to track detected markers and whether they have been logged

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)

    def callback_pose(self, msg_in):
        self.current_location = msg_in.pose.position

    def img_callback(self, msg_in):
        if msg_in.header.stamp > self.last_msg_time:
            try:
                frame = self.br.compressed_imgmsg_to_cv2(msg_in)
            except CvBridgeError as e:
                rospy.logerr(e)

            aruco = self.find_aruco(frame)

            # Publish the averaged location if the marker has not been detected for a while
            current_time = time.time()
            if not self.aruco_detected and len(self.detected_positions) > 0:
                if current_time - self.last_seen_time >= self.detection_timeout:
                    self.publish_average_location()
                    self.detected_positions.clear()  # Clear the list after publishing

            self.publish_to_ros(aruco)
            self.aruco_detected = False
            self.time_finished_processing = rospy.Time.now()

    def aruco_frame_translation(self, camera_location):
        world_x = self.current_location.x
        world_y = self.current_location.y
        world_z = self.current_location.z
        uav_location = [world_x, world_y, world_z]

        camera_offset_x = (208 - camera_location[0]) / 208 #208
        camera_offset_y = (208 - camera_location[1]) / 208

        offset_x = camera_offset_x * world_z * tan(self.camera_FOV_x / 2) 
        offset_y = camera_offset_y * world_z * tan(self.camera_FOV_y / 2) 

        if (offset_x > 0 and offset_y < 0) or (offset_x < 0 and offset_y > 0):
            offset_y = -offset_y
        else:
            offset_x = -offset_x

        world_x += offset_x
        world_y += offset_y 

        world_location = [world_x, world_y, world_z]
        return world_location, uav_location

    def find_aruco(self, frame):
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            ids = ids.flatten()

            for (marker_corner, marker_ID) in zip(corners, ids):
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # Log message and publish ID once per detection
                if marker_ID not in self.detected_markers:
                    self.pub_aruco_vocal.publish(marker_ID)
                    rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))
                    self.detected_markers[marker_ID] = True

                if marker_ID == self.desired_aruco_id:
                    frame_x = (top_left[0] + bottom_right[0]) / 2
                    frame_y = (top_left[1] + bottom_right[1]) / 2
                    aruco_location, _ = self.aruco_frame_translation([frame_x, frame_y])

                    # Store detected position for averaging
                    self.detected_positions.append(aruco_location)
                    self.aruco_detected = True
                    self.last_seen_time = time.time()

                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                cv2.putText(frame, str(marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def publish_average_location(self):
        if len(self.detected_positions) == 0:
            return

        avg_location = np.mean(self.detected_positions, axis=0)
        msg_out = ArucoLocalisation()
        msg_out.frame_x = avg_location[0]
        msg_out.frame_y = avg_location[1]
        msg_out.aruco_id = self.desired_aruco_id

        rospy.loginfo(f'Desired Aruco: [{self.desired_aruco_id}] Found. Aruco Location at x: {avg_location[0]}, y: {avg_location[1]}')
        self.aruco_pub_inf.publish(msg_out)

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()

        self.aruco_pub.publish(msg_out)

def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()