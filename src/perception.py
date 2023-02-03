#!/usr/bin/python3

import cv2
import math
import time
import pyrealsense2 as rs2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import CameraInfo, Image
from localization_project.msg import perception_data
from vision_file.utils import ARUCO_DICT, aruco_display
from visualization_msgs.msg import Marker, MarkerArray

class Perception:
    def __init__(self):
        self.frames_counter=0
        self.bridge = CvBridge()
        # PUBLISHERS
        self.perception_pub = rospy.Publisher("/preception_topic", perception_data,queue_size=2)
        self.image_pub = rospy.Publisher("/image_topic_2",Image,queue_size=2)
        self.marker_pub = rospy.Publisher("/realsense/markers", MarkerArray, queue_size=2)

        # SUBSCRIBERS
        # Subscriber to the realsense camera
        self.image_sub = rospy.Subscriber('/turtlebot/realsense_d435i/color/image_raw', Image,self.get_image) 
        self.depth_sub = rospy.Subscriber("/turtlebot/realsense_d435i/depth/image_raw", Image, self.imageDepthCallback)
        self.depth_sub_info = rospy.Subscriber("/turtlebot/realsense_d435i/depth/camera_info", CameraInfo, self.imageDepthInfoCallback)
       
        self.dic={}
        self.intrinsics = None
        self.flag= False
        self.depth=[]
    
    def get_image(self,image):
        self.frames_counter+=1
        if self.frames_counter==6:
            print("frame counter=6")
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image,'bgr8')
                arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT['DICT_4X4_50'])
                arucoParams = cv2.aruco.DetectorParameters_create()
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, arucoDict, parameters=arucoParams)
                detected_markers = aruco_display(corners, ids, rejected, cv_image)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                if ids is not None:
                    for idx, id in enumerate(ids): 
                        # print(id)
                        x_sum = corners[idx][0][0][0]+ corners[idx][0][1][0]+ corners[idx][0][2][0]+ corners[idx][0][3][0]
                        y_sum = corners[idx][0][0][1]+ corners[idx][0][1][1]+ corners[idx][0][2][1]+ corners[idx][0][3][1]
                        x_centerPixel = int(round(x_sum*.25,0))
                        y_centerPixel = int(round(y_sum*.25,0))
                        center=[y_centerPixel,x_centerPixel]
                        # print ("id and location",id , center)
                        self.dic[id[0]]=center
                self.flag= True
                if self.flag==True and self.dic is not None:
                    # print("DEPTH",self.depth)
                    # if self.depth is not None:
                    # print("byeeeeeeeee")
                    print("dic",self.dic)
                    self.localization_conversions()
                    self.frames_counter=0
                    self.dic={}
                    # self.depth=[]
                    # print(self.dic)
            except CvBridgeError as e:
                print(e)
    
    def imageDepthCallback(self, data):
        # print(self.flag)
        # self.depth=[]
        if self.flag==True and self.dic is not None:
            # print("dictionary",self.dic)
            keys=list(self.dic)
            for idx, id in enumerate(keys):
                x_pos=self.dic[keys[idx]][0]
                y_pos=self.dic[keys[idx]][1]
                # print("pos ", x_pos, " ", y_pos)
                cv_image = self.bridge.imgmsg_to_cv2(data,data.encoding)
                # print(cv_image.shape)
                depth_computed= cv_image[x_pos,y_pos]
                # print("depth_computed",depth_computed)
                self.depth.append(depth_computed)
        # self.depth=[]
        # print("depth list",self.depth)
        
    
    def imageDepthInfoCallback(self, cameraInfo):
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
           
    
    def localization_conversions(self):
        # print("depth ",self.depth)
        time.sleep(3)
        ma = MarkerArray()
        if self.depth is not None:
            if self.dic is not None:
                # print("A")
                keys=list(self.dic)
                for idx, id in enumerate(keys):
                    print("id",id)
                    y_pix=self.dic[keys[idx]][0]
                    x_pix=self.dic[keys[idx]][1]
                    # x_meters=(x_pix-self.intrinsics.ppx)/self.intrinsics.fx
                    # y_meters=(y_pix-self.intrinsics.ppy)/self.intrinsics.fx

                    x_pixels=(x_pix-self.intrinsics.ppx)
                    y_pixels=(y_pix-self.intrinsics.ppy)

                    print("focal length",self.intrinsics.fx)
                    print("x_pixels",x_pixels)
                    print("y_pixels",y_pixels)
                    print("depth",self.depth[idx])
                    theta=math.atan2(x_pixels,self.intrinsics.fx)
            
                    # rho_projected=(self.depth)*math.cos(beta)
                    print("theta",theta)
                    # print("rho_projected",rho_projected)
                    msg = perception_data()
                    msg.rho_projected=self.depth[idx]
                    msg.theta=theta
                    msg.class_id=id
                    self.perception_pub.publish(msg)
                    print("marker location x",self.depth[idx] * math.cos(theta))
                    print("marker location y",self.depth[idx] * math.sin(theta))
            
                    m = Marker()
                    m.id = id
                    m.header.stamp = rospy.Time.now()
                    m.header.frame_id = "realsense_link"
                    m.type = m.CUBE
                    m.pose.position.z = self.depth[idx] * math.cos(theta)
                    m.pose.position.x =  self.depth[idx] * math.sin(theta)
                    m.pose.position.y = 0.0

                    m.pose.orientation.w = 1.0

                    m.color.a = 1.0
                    m.color.r = 1.0

                    m.scale.x = 0.1
                    m.scale.y = 0.1
                    m.scale.z = 0.1

                    ma.markers.append(m)
            self.depth=[] 
        self.marker_pub.publish(ma) 
                      




# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('Vision_node') 
    node = Perception()
    rospy.spin()
    rospy.Rate(10)
        
