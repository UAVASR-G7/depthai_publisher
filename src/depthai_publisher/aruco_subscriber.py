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


class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'

    def __init__(self):
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        
        # Callback to save "current location" such that we can perform and return from a diversion to the correct location
        #self.sub_pose = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.callback_pose) # For flight
        self.sub_pose = rospy.Subscriber("uavasr/pose", PoseStamped, self.callback_pose) # Use for emulator
        
        self.aruco_pub_inf = rospy.Publisher('/processed_aruco/localisation', ArucoLocalisation, queue_size=10)
        self.br = CvBridge()
        self.last_msg_time = rospy.Time(0)
        self.lock = threading.Lock()

        # Camera Variables
        self.camera_FOV_x = 54 * (pi / 180) # [rad]
        self.camera_FOV_y = 66 * (pi / 180) # [rad]

        # Pose
        self.current_location = Point()

        # Published for GCS VOCAL
        # self.pub_land_vocal = rospy.Publisher('vocal/land', Bool, queue_size = 10)
        self.pub_aruco_vocal = rospy.Publisher('vocal/aruco', Int32, queue_size = 10)
        # self.pub_target_vocal = rospy.Publisher('payload/target', TargetLocalisation, queue_size = 10)

        # Aruco Variables
        self.desired_aruco_id = 72 # Changed to the aruco marker to land
        self.previous_aruco_id = -1
        self.FoundAruco = False # If aruco is not found, land at origin (aruco landing contigency)

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)

    # This function will check receive the current pose of the UAV constantly
    def callback_pose(self, msg_in):
        #rospy.loginfo("Pose Callback recieved/Triggered")
        # Store the current position at all times so it can be accessed later
        self.current_location = msg_in.pose.position

    def img_callback(self, msg_in):
        if msg_in.header.stamp > self.last_msg_time:
            try:
                frame = self.br.compressed_imgmsg_to_cv2(msg_in)
            except CvBridgeError as e:
                rospy.logerr(e)

            aruco = self.find_aruco(frame)
            self.publish_to_ros(aruco)
            # self.publish_marker(aruco)

            self.time_finished_processing = rospy.Time.now()
	
    # This function is used to translate between the camera frame and the world location when undertaking aruco detection
    def aruco_frame_translation(self, camera_location):
        # The initial location of the UAV
        world_x = self.current_location.x
        world_y = self.current_location.y
        world_z = self.current_location.z
        uav_location = [world_x, world_y, world_z]

        # Normalised position of the target within the camera frame [-1, 1] in both x- and y-directions
        # Positive values correspond to positive values in the world frame
        # The input camera location is given as the pixel position of the aruco centroid within the frame
        camera_offset_x = (208 - camera_location[0]) / 208
        camera_offset_y = (208 - camera_location[1]) / 208

        # The offset from the UAV of the target, based on the location within the camera frame
        offset_x = camera_offset_x * world_z * tan(self.camera_FOV_x / 2) 
        offset_y = camera_offset_y * world_z * tan(self.camera_FOV_y / 2) 

        # To make sure that it follows the same axis orientation as the uav
        if (offset_x > 0 and offset_y < 0) or (offset_x < 0 and offset_y > 0):
            offset_y = -offset_y
        else:
            offset_x = -offset_x

        # Add the offset to the initial location to determine the target location
        world_x += offset_x
        world_y += offset_y

        # Store the world location in a single array to be returned by the function
        world_location = [world_x, world_y, world_z]
        return world_location, uav_location

    def find_aruco(self, frame):
        msg_out = ArucoLocalisation()

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

                # self.current_aruco_id = marker_ID
                if marker_ID != self.previous_aruco_id:
                    self.pub_aruco_vocal.publish(marker_ID)
                    rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))
                    if self.desired_aruco_id == marker_ID and not self.FoundAruco:
                        frame_x = (top_left[0] + bottom_right[0]) / 2
                        frame_y = (top_left[1] + bottom_right[1]) / 2
                        aruco_location, uav_location = self.aruco_frame_translation([frame_x, frame_y])
                        rospy.loginfo(f'UAV Location at x: {uav_location[0]}, y: {uav_location[1]}, z: {uav_location[2]}')
                        rospy.loginfo(f'Desired Aruco Marker {marker_ID} Detected at x: {aruco_location[0]}, y: {aruco_location[1]}')
                        msg_out.frame_x = aruco_location[0]
                        msg_out.frame_y = aruco_location[1]
                        msg_out.aruco_id = marker_ID
                        self.aruco_pub_inf.publish(msg_out)
                        self.FoundAruco = True
                    self.previous_aruco_id = marker_ID

                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                cv2.putText(frame, str(
                    marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

        self.aruco_pub.publish(msg_out)


def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()
