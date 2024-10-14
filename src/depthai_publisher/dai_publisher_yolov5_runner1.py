#!/usr/bin/env python3

'''
Run as:
# check model path line ~30is
rosrun depthai_publisher dai_publisher_yolov5_runner
'''
############################### ############################### Libraries ###############################
from pathlib import Path
import threading
import csv
import argparse
import time
import sys
import json     # Yolo conf use json files
import cv2
import numpy as np
import depthai as dai
import rospy
import tf2_ros
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from spar_msgs.msg import TargetLocalisation
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from std_msgs.msg import Time
from math import *

############################### ############################### Parameters ###############################
# Global variables to deal with pipeline creation
pipeline = None
cam_source = 'rgb' #'rgb', 'left', 'right'
cam=None
# sync outputs
syncNN = True
# model path
modelsPath = "/home/cdrone/catkin_ws/src/depthai_publisher/src/depthai_publisher/models"
# modelName = 'exp31Yolov5_ov21.4_6sh'
modelName = 'mission1v2'
# confJson = 'exp31Yolov5.json'
confJson = 'best.json'

################################  Yolo Config File
# parse config
configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# Extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
# Parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

class DepthaiCamera():
    # res = [416, 416]
    fps = 20.0

    pub_topic = '/depthai_node/image/compressed'
    pub_topic_raw = '/depthai_node/image/raw'
    pub_topic_detect = '/depthai_node/detection/compressed'
    pub_topic_cam_inf = '/depthai_node/camera/camera_info'

    def __init__(self):
        self.pipeline = dai.Pipeline()

         # Input image size
        if "input_size" in nnConfig:
            self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

        # Publishers for target
        self.target_pub_inf = rospy.Publisher("target_detection/localisation", TargetLocalisation, queue_size=10)
        self.pub_found = rospy.Publisher('/emulated_uav/target_found', Time, queue_size=10)
        
        # Callback to save "current location" such that we can perform and return from a diversion to the correct location
        # self.sub_pose = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.callback_pose) # Use for flight
        self.sub_pose = rospy.Subscriber("uavasr/pose", PoseStamped, self.callback_pose) # Use for emulator

        # Pose
        self.current_location = Point()

        # Target confidence
        self.target_confidence_threshold = 0.9
        
        # Variables for the target
        self.first_target = False
        self.second_target = False

        # Camera Variables
        self.camera_FOV_x = 54 * (pi / 180) # [rad]
        self.camera_FOV_y = 66 * (pi / 180) # [rad]

        # Setup tf2 broadcaster and timestamp publisher
        self.tfbr = tf2_ros.TransformBroadcaster()

        # Pulbish ros image data
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
        # Create a publisher for the CameraInfo topic
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=10)
        # Create a timer for the callback
        self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)

        rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

        self.br = CvBridge()

        rospy.on_shutdown(lambda: self.shutdown())

    # This function will check receive the current pose of the UAV constantly
    def callback_pose(self, msg_in):
        #rospy.loginfo("Pose Callback recieved/Triggered")
        # Store the current position at all times so it can be accessed later
        self.current_location = msg_in.pose.position

    def publish_camera_info(self, timer=None):
        # Create a publisher for the CameraInfo topic

        # Create a CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.height = self.nn_shape_h # Set the height of the camera image
        camera_info_msg.width = self.nn_shape_w  # Set the width of the camera image

        # Set the camera intrinsic matrix (fx, fy, cx, cy)
        # camera_info_msg.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0] # old
        camera_info_msg.K = [619.994, 0.0, 217.039, 0.0, 620.141, 199.928, 0.0, 0.0, 1.0]

        # Set the distortion parameters (k1, k2, p1, p2, k3)
        # camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204] # old
        camera_info_msg.D = [0.134107, -0.71825, -0.00575186, 0.00481137, 0.0]

        # Set the rectification matrix (identity matrix)
        # camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] # old
        camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Set the projection matrix (P)
        # camera_info_msg.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0] # old
        camera_info_msg.P = [622.481, 0.0, 217.663, 0.0, 0.0, 622.491, 198.122, 0.0, 0.0, 0.0, 1.0, 0.0]

        # Set the distortion model
        camera_info_msg.distortion_model = "plumb_bob"
        # Set the timestamp
        camera_info_msg.header.stamp = rospy.Time.now()

        self.pub_cam_inf.publish(camera_info_msg)  # Publish the camera info message

    def rgb_camera(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(self.res[0], self.res[1])
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(self.fps)

        # Def xout / xin
        ctrl_in = self.pipeline.createXLinkIn()
        ctrl_in.setStreamName("cam_ctrl")
        ctrl_in.out.link(cam_rgb.inputControl)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("video")

        cam_rgb.preview.link(xout_rgb.input)

    def target_offset(self, camera_location): ### NEED TO CLEAN THIS FUNCTION
        # The initial location of the UAV 
        world_z = self.current_location.z

        # Normalised position of the target within the camera frame [-1, 1] in both x- and y-directions
        # Positive values correspond to positive values in the world frame
        # The input camera location is given as the pixel position of the aruco centroid within the frame
        camera_offset_x = (0.5 - camera_location[0]) / 0.5
        camera_offset_y = (0.5 - camera_location[1]) / 0.5

        # rospy.loginfo(f'offset_x: {camera_offset_x}, offset_y: {camera_offset_y}!')

        # The offset from the UAV of the target, based on the location within the camera frame
        offset_x = camera_offset_x * world_z * tan(self.camera_FOV_x / 2) 
        offset_y = camera_offset_y * world_z * tan(self.camera_FOV_y / 2) 

        # rospy.loginfo(f'offset_x: {offset_x}, offset_y: {offset_y}!')

        return [offset_x, offset_y]
    
    # This function is used to translate between the camera frame and the world location when undertaking aruco detection
    def target_world_location(self, camera_location):
        # Add the offset to the initial location to determine the target location
        world_x = self.current_location.x
        world_y = self.current_location.y

        offsets = self.target_offset(camera_location)

        world_x += offsets[1]
        world_y += offsets[0]

        # Store the world location in a single array to be returned by the function
        world_location = [world_x - 0.10, world_y]
        return world_location, offsets
    
    def process_target_info(self, detection):
        # Initialise
        msg_out_localisation = TargetLocalisation()
        msg_out_tf = TransformStamped()
        time_found = rospy.Time.now()
        world_z = self.current_location.z

        # Calculate location of target
        frame_x = (detection.xmin + detection.xmax) / 2
        frame_y = (detection.ymin + detection.ymax) / 2
        # target_offsets = self.target_offset([frame_x, frame_y])
        location, target_offsets = self.target_world_location([frame_x, frame_y])

        # Localisation msg
        msg_out_localisation.target_label = labels[detection.label]
        msg_out_localisation.target_id = detection.label
        msg_out_localisation.frame_x = location[0]
        msg_out_localisation.frame_y = location[1]
        self.target_pub_inf.publish(msg_out_localisation)

        # TF msg
        msg_out_tf.header.stamp = time_found
        msg_out_tf.header.frame_id = "camera"
        msg_out_tf.child_frame_id = "target"
        
        msg_out_tf.transform.translation.x = - target_offsets[0] + 0.10
        msg_out_tf.transform.translation.y = target_offsets[1] 
        msg_out_tf.transform.translation.z = world_z - 0.15
        msg_out_tf.transform.rotation.x = 0
        msg_out_tf.transform.rotation.z = 0
        msg_out_tf.transform.rotation.y = 0
        msg_out_tf.transform.rotation.w = 1.0
        self.tfbr.sendTransform(msg_out_tf)
        self.pub_found.publish(time_found)

        rospy.loginfo(f'Target [{labels[detection.label]}] Detected')

    def run(self):
        #self.rgb_camera()
        ############################### Run Model ###############################
        # Pipeline defined, now the device is assigned and pipeline is started
        pipeline = None
        # Get argument first
        # Model parameters
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
        print(metadata)
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
        print(nnPath)

        pipeline = self.createPipeline(nnPath)

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            # rospy.loginfo(device.getConnectedCameraFeatures())
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            if cam_source != "rgb" and not depth_enabled:
                raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
            device.startPipeline(pipeline)

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            start_time = time.time()
            counter = 0
            fps = 0
            
            olor2 = (255, 255, 255)
            layer_info_printed = False
            dims = None

            while True:
                found_classes = []
                # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
                inRgb = q_nn_input.get()
                inDet = q_nn.get()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                else:
                    print("Cam Image empty, trying again...")
                    continue
                
                if inDet is not None:
                    detections = inDet.detections
                    # print(detections)
                    for detection in detections:
                        # print(detection)
                        # print("{},{},{},{},{},{},{}".format(detection.label,labels[detection.label],detection.confidence,detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        # rospy.loginfo(f'Target [{labels[detection.label]}] Found at x-min: {detection.xmin} x-max: {detection.xmax}, y-min: {detection.ymin} y-max: {detection.ymax}')
                        found_classes.append(detection.label)

                        if self.current_location.z > 2: # start detection at 1.5
                            if detection.confidence > self.target_confidence_threshold:
                                #rospy.loginfo(f"Confidence:{detection.confidence}")
                                if detection.label == 0 and not self.first_target: # backpack
                                    self.process_target_info(detection)
                                    self.first_target = True
                                elif detection.label == 1 and not self.second_target: # person
                                    self.process_target_info(detection)
                                    self.second_target = True

                        #print(dai.ImgDetection.getData(detection))
                    found_classes = np.unique(found_classes)
                    # print(found_classes)
                    overlay = self.show_yolo(frame, detections)
                else:
                    print("Detection empty, trying again...")

                    continue

                if frame is not None:
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    # cv2.imshow("nn_output_yolo", overlay)
                    self.publish_to_ros(frame)
                    self.publish_detect_to_ros(overlay)
                    self.publish_camera_info()

                ## Function to compute FPS
                counter+=1
                if (time.time() - start_time) > 1 :
                    fps = counter / (time.time() - start_time)

                    counter = 0
                    start_time = time.time()


            # with dai.Device(self.pipeline) as device:
            #     video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

            #     while True:
            #         frame = video.get().getCvFrame()

            #         self.publish_to_ros(frame)
            #         self.publish_camera_info()

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image.publish(msg_out)
        # Publish image raw
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_image_raw.publish(msg_img_raw)

    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image_detect.publish(msg_out)
        
    ############################### ############################### Functions ###############################
    ######### Functions for Yolo Decoding
    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def show_yolo(self, frame, detections):
        color = (255, 0, 0)
        # Both YoloDetectionNetwork and MobileNetDetectionNetwork output this message. This message contains a list of detections, which contains label, confidence, and the bounding box information (xmin, ymin, xmax, ymax).
        overlay =  frame.copy()
        for detection in detections:
            bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(overlay, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return overlay

    # Start defining a pipeline
    def createPipeline(self, nnPath):

        pipeline = dai.Pipeline()

        # pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)
        # pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        # Network specific settings
        detection_nn.setConfidenceThreshold(confidenceThreshold)
        detection_nn.setNumClasses(classes)
        detection_nn.setCoordinateSize(coordinates)
        detection_nn.setAnchors(anchors)
        detection_nn.setAnchorMasks(anchorMasks)
        detection_nn.setIouThreshold(iouThreshold)
        # generic nn configs
        detection_nn.setBlobPath(nnPath)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        # Define a source - color camera
        if cam_source == 'rgb':
            cam = pipeline.create(dai.node.ColorCamera)
            cam.initialControl.setManualFocus(120)
            cam.setPreviewSize(self.nn_shape_w,self.nn_shape_h)
            cam.setInterleaved(False)
            cam.preview.link(detection_nn.input)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(35.0)
            print("Using RGB camera...")
        elif cam_source == 'left':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            print("Using BW Left cam")
        elif cam_source == 'right':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            print("Using BW Rigth cam")

        if cam_source != 'rgb':
            manip = pipeline.create(dai.node.ImageManip)
            manip.setResize(self.nn_shape_w,self.nn_shape_h)
            manip.setKeepAspectRatio(True)
            # manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
            manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
            cam.out.link(manip.inputImage)
            manip.out.link(detection_nn.input)

        # Create outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("nn_input")
        xout_rgb.input.setBlocking(False)

        detection_nn.passthrough.link(xout_rgb.input)

        xinDet = pipeline.create(dai.node.XLinkOut)
        xinDet.setStreamName("nn")
        xinDet.input.setBlocking(False)

        detection_nn.out.link(xinDet.input)

        return pipeline


    def shutdown(self):
        cv2.destroyAllWindows()


#### Main code that creates a depthaiCamera class and run it.
def main():
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()

    while not rospy.is_shutdown():
        dai_cam.run()

    dai_cam.shutdown()
