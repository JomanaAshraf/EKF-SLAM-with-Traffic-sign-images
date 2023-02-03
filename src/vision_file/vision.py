#!/usr/bin/python3
import torch
import math
import time
from torchvision import  transforms
from collections import  OrderedDict
import pandas as pd
from itertools import product as product
from PIL import Image as PILImage
from sensor_msgs.msg import CameraInfo, Image
from model import *
import torch
import rospy
from my_eval_sim import my_evaluate
import cv2
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs2
from localization_project.msg import perception_data


class Prediction:
    def __init__(self ):

        self.frames_counter=0
        self.bridge = CvBridge()
        # PUBLISHERS
        self.perception_pub = rospy.Publisher("/preception_topic", perception_data,queue_size=1)
            
        # SUBSCRIBERS
        # Subscriber to the realsense camera
        self.image_sub = rospy.Subscriber('/turtlebot/realsense_d435i/color/image_raw', Image,self.get_image) 
        self.depth_sub = rospy.Subscriber("/turtlebot/realsense_d435i/depth/image_raw", Image, self.imageDepthCallback)
        self.depth_sub_info = rospy.Subscriber("/turtlebot/realsense_d435i/depth/camera_info", CameraInfo, self.imageDepthInfoCallback)
       
        self.dic={}
        self.intrinsics = None
        self.flag= False
        self.depth=None
       
    def get_image(self,image):
        self.frames_counter+=1
        if self.frames_counter==6:
            print("frame counter=6")
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image,'bgr8')
                cv_image=cv2.resize(cv_image,(600,600))
                print('new image received')
            except CvBridgeError as e:
                print(e) 
            im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.prediction(im_rgb)
            print("beforeeeee",self.dic)
            print(self.flag)
            print("after",self.dic)
            if self.flag==True and self.dic is not None:
                # print("DEPTH",self.depth)
                # if self.depth is not None:
                print("byeeeeeeeee")
                theta,rho_projected=self.localization_conversions()
                self.publish_for_localization(theta,rho_projected)
                self.frames_counter=0
                self.dic={}
                print(self.dic)

            

    def prediction(self,image):
        self.flag= False
        # Prepare Label-to-int & Int-to-label Dictionaries
        all_labels_df=pd.read_csv("/home/jomana/masters_ws/src/Studying/Semester_2/Hands_on_Localization/localization_project/src/vision_file/allAnnotations.csv",sep=';')
        all_labels_df['Annotation tag'].value_counts()
        # print(set(all_labels_df['Annotation tag']))
        label_to_int = OrderedDict({label: num for num, label in enumerate(set(all_labels_df['Annotation tag']), start=1)})
        label_to_int['background'] = 0
        int_to_label = {v: k for k, v in label_to_int .items()}
        int_to_label= OrderedDict(sorted(int_to_label.items(), key=lambda t: t[0]))
        #Good formatting when printing the APs for each class and mAP
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SSD600(len(label_to_int))
        # Load model checkpoint
        checkpoint = '/home/jomana/masters_ws/src/Studying/Semester_2/Hands_on_Localization/localization_project/src/vision_file/checkpoint.pth.tar'
        checkpoint = torch.load(checkpoint,map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        # print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        state_dict = checkpoint['model'].state_dict()
        model.load_state_dict(state_dict )
        model = model.to(device)
        model.eval()

        # Transforms
        resize = transforms.Resize((600,600))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
     
        # Evaluate and annotate
        original_image = PILImage.fromarray(image)    
        self.dic=my_evaluate(device,model,int_to_label,normalize,resize,to_tensor,original_image, annotate_image=True)
        print("dic",self.dic)
        self.flag= True

    def imageDepthCallback(self, data):
        # print(self.flag)
        if self.flag==True and self.dic is not None:
            # print("inside flag")
            keys=list(self.dic)
            x_pos=int(round(self.dic[keys[0]][0],0))
            y_pos=int(round(self.dic[keys[0]][1],0))
            # print(x_pos)
            cv_image = self.bridge.imgmsg_to_cv2(data,data.encoding)
            self.depth = cv_image[x_pos,y_pos]
            # print("returned depth",self.depth)


    def imageDepthInfoCallback(self, cameraInfo):
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
           
    
    def localization_conversions(self):
        print("depth ",self.depth)
        time.sleep(3)
        if self.depth is not None:
            if self.dic is not None:
                print("A")
                keys=list(self.dic)
                x_pix=self.dic[keys[0]][0]
                y_pix=self.dic[keys[0]][1]
                x_meters=(x_pix-self.intrinsics.ppx)/self.intrinsics.fx
                y_meters=(y_pix-self.intrinsics.ppy)/self.intrinsics.fx
                print("focal length",self.intrinsics.fx)
                print("x_meters",x_meters)
                print("y_meters",y_meters)
                print("depth",self.depth)
                theta=math.atan2(x_meters,self.intrinsics.fx)
                hypot=math.sqrt(x_meters**2+self.intrinsics.fx**2)
                beta=math.atan2(hypot,y_meters)
                rho_projected=(self.depth*0.001)*math.cos(beta)

                print("theta",theta)
                print("rho_projected",rho_projected)
                return theta,rho_projected
        
    def publish_for_localization(self, theta,rho_projected):
        print("in publish_for_localization")
        keys=list(self.dic.keys())
        msg = perception_data()
        msg.rho_projected=rho_projected
        msg.theta=theta
        msg.class_id=keys[0]
    
        self.perception_pub.publish(msg)


# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('Vision_node') 
    node = Prediction()
    rospy.spin()
        