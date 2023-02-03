#!/usr/bin/env python

import rospy
import math
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from localization_project.msg import perception_data
import numpy as np
import slam_functions as func
from ekf_slam import EKFSlam
# import pygame
# import sys

class LocalizationNode(EKFSlam):
    def __init__(self, xr_init, Pr_init,odom_topic,detected_sign_topic,initial_map = None):
        super().__init__(xr_init, Pr_init, 100, 2, initial_map)

        # Publishers
        self.pub_predicted_position = rospy.Publisher("predicted_position", Odometry,queue_size=1)
        self.pub_uncertainity = rospy.Publisher("uncertainity", Marker,queue_size=2)

        # Subscribers
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        self.perception_sub = rospy.Subscriber(detected_sign_topic, perception_data, self.perception_callback)


        # Incremental odometry
        self.last_odom = None
        self.odom = None

        self.traffic_signs = {}

        # Times
        self.time = rospy.Time(0)
        self.odomtime = rospy.Time(0)

        #Noises
        self.Q = np.array([[0.25**2, 0,0],
                            [0, 0.25**2,0],
                            [0,0,0.25**2]])
        # Measurement noise
        self.R = np.array([[0.2**2, 0],
                            [0, 0.2**2]])

        # Flags
        self.uk = None
        self.pub = False
        self.new_odom = False
        self.camera = False
        self.actual_class_id=None

        # Filter
        self.ekf = EKFSlam(xr_init, Pr_init, 100, 2, initial_map)

        self.ground_truth={"dip":[4.65,-0.83],"stop":[3.07,-5.5],"signalAhead":[-5.773094,-5.492466],
                            "noLeftTurn":[-7.75,-1.559],"schoolSpeedLimit25":[-4.427,-1.434],
                            "Reverse Turn":[1.25,1.259],"stop":[7.937017,3.08031],
                            "stop":[4.763,-0.83]}

    def odom_callback(self, msg):
        """Publish tf and calculate incremental odometry."""
        # Save time
        self.odomtime = msg.header.stamp
        self.odom = msg

        # Incremental odometry
        if self.last_odom is not None:
            # Increment computation
            delta_x = msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
            delta_y = msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
            self.yaw = func.yaw_from_quaternion(msg.pose.pose.orientation)
            self.previous_yaw = func.yaw_from_quaternion(self.last_odom.pose.pose.orientation)
            # print("yaw",yaw)
            # print("previous yaw",previous_yaw)
            # Odometry seen from vehicle frame
            self.uk = np.array([delta_x * np.cos(self.previous_yaw) +
                                delta_y * np.sin(self.previous_yaw),
                                -delta_x * np.sin(self.previous_yaw) +
                                delta_y * np.cos(self.previous_yaw),
                                func.angle_wrap(self.yaw - self.previous_yaw)])

            # Adding noise to the control
            self.uk=self.uk*0.001*np.random.rand(3,)
            # Flag available
            self.new_odom = True

        # Save initial odometry for increment
        else:
            print("no new odom")
            self.last_odom = msg
    

    def perception_callback(self,msg):
        # print("in preception")
        self.rho = msg.rho_projected
        self.theta=msg.theta
        self.class_id=msg.class_id

        self.z=np.array([self.rho,self.theta])
        self.xf=np.array([self.rho*math.sin(self.ekf.x_[2])*math.sin(self.theta)+self.rho*math.cos(self.ekf.x_[2])*math.cos(self.theta),
                        -self.rho*math.cos(self.ekf.x_[2])*math.sin(self.theta)+self.rho*math.sin(self.ekf.x_[2])*math.cos(self.theta)])
        # Flag
        self.camera = True


    def get_ekf_msgs(self, ekf):
        """
        Create messages to visualize EKFs.

        The messages are odometry and uncertainity.

        :param EKF ekf: the EKF filter.
        :returns: a list of messages containing the odometry of the filter and the uncertainty
        """
        # print("in get ekf")
        # Time
        time = rospy.Time.now()

        # Odometry
        # print("hiiiiiiiiiiiiii")
        msg_odom = Odometry()
        msg_odom.header.stamp = time
        msg_odom.header.frame_id = 'base_link'
        # print("ekf",ekf.x_[0:3])
        msg_odom.pose.pose.position.x = ekf.x_[0]
        msg_odom.pose.pose.position.y = ekf.x_[1]
        msg_odom.pose.pose.position.z = 0
        quat = func.quaternion_from_yaw(ekf.x_[2])
        msg_odom.pose.pose.orientation.x = quat[0]
        msg_odom.pose.pose.orientation.y = quat[1]
        msg_odom.pose.pose.orientation.z = quat[2]
        msg_odom.pose.pose.orientation.w = quat[3]
        msg_odom.pose.covariance=[ekf.P_[0,0],ekf.P_[0,1],0,0,0,ekf.P_[0,2],ekf.P_[1,0],ekf.P_[1,1],0,0,0,ekf.P_[1,2]
                            ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,ekf.P_[2,0],ekf.P_[2,1],0,0,0,ekf.P_[2,2]]

        # Uncertainity
        # print("byeeeeeeeeeeeeeeee")
        uncert = ekf.P_[0:3, 0:3].copy()
        val, vec = np.linalg.eigh(uncert)
        yaw = np.arctan2(vec[1, 0], vec[0, 0])
        quat = func.quaternion_from_yaw(yaw)
        msg_ellipse = Marker()
        msg_ellipse.header.frame_id = "base_link"
        msg_ellipse.header.stamp = time
        msg_ellipse.type = Marker.CYLINDER
        msg_ellipse.pose.position.x = ekf.x_[0]
        msg_ellipse.pose.position.y = ekf.x_[1]
        msg_ellipse.pose.position.z = -0.1  
        msg_ellipse.pose.orientation.x = quat[0]
        msg_ellipse.pose.orientation.y = quat[1]
        msg_ellipse.pose.orientation.z = quat[2]
        msg_ellipse.pose.orientation.w = quat[3]
        msg_ellipse.scale.x = math.sqrt(val[0])
        msg_ellipse.scale.y = math.sqrt(val[1])
        msg_ellipse.scale.z = 0.05
        msg_ellipse.color.a = 1
        msg_ellipse.color.r = 0.0
        msg_ellipse.color.g = 1
        msg_ellipse.color.b = 0.0

        return msg_odom, msg_ellipse

    
    def publish_results(self):
        """Publishe results from the filter."""

        # Get filter data
        predicted_odom, ellipse = self.get_ekf_msgs(self.ekf)

        # Publish results
        self.pub_predicted_position.publish(predicted_odom)
        self.pub_uncertainity.publish(ellipse)

        # # Configuration
        # pygame.init()
        # fps = 60
        # fpsClock = pygame.time.Clock()
        # width, height = 800, 200
        # screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

        # while  self.actual_class_id is not None:

        #     # System Font
        #     font = pygame.font.SysFont('Garamond', 30)
        #     text="The predicted class is wrong, the right one is: " + str(self.actual_class_id)
        #     textsurface = font.render(text, False, (200, 200, 200))

        # # Game loop.
        #     screen.fill((20, 20, 20))
        #     # for event in pygame.event.get():
        #     #     if event.type == pygame.QUIT:
        #     #         pygame.quit()
        #     #         sys.exit()

        #     screen.blit(textsurface,(80, 100))

        #     pygame.display.flip()
        #     fpsClock.tick(fps)

    def iterate(self):
        """Main loop of the filter."""
        # Prediction
        if self.new_odom:
            # print("in prediction")
            check_rotation=math.isclose(self.yaw,self.previous_yaw,abs_tol=0.01)
            # print("checkkkkkkkkkkkkk",check_rotation)
            if self.uk[0]!=0 or self.uk[1]!=0 or check_rotation == False:
                [xr, pr, prm] = self.ekf.prediction(self.uk,self.Q)
                self.ekf.applyPrediction(xr,pr,prm)
                self.last_odom = self.odom  # new start odom for incremental
                self.new_odom = False
                self.pub=True
                self.time = self.odomtime

        if self.camera:
            # Perform data association to know the actual class_id
            xr = self.ekf.x_[0:3]
            z=func.h(xr,self.xf)
            hyp = func.data_association(self.ground_truth,xr,self.ekf.P_[0:3,0:3],z,self.R)
            keys=list(self.ground_truth)
            self.actual_class_id = keys[hyp]
            print("actual class is: ",self.actual_class_id)
            # TODO: RETURN TO THE PERCEPTION MODEL THE ACTUAL ID TO ENHANCE IT

            # Known landmark! Do Upate
            if self.actual_class_id in self.traffic_signs.keys():
                idx = self.traffic_signs[self.actual_class_id][2]
                xf = self.xf
                [x, P] = self.update(self.z,
                                    self.R,
                                    idx,
                                    func.h(xr,xf), 
                                    func.Jhxr(xr, xf),
                                    func.Jhxf(xr, xf),
                                    func.Jhv())
                self.applyUpdate(x, P)
                self.camera = False 
                self.pub=True 

            #Unknown landmark! Do initialization
            else: 
                idx = self.add_landmark(func.g(xr,self.z), 
                                        func.Jgxr(xr,self.z), 
                                        func.Jgz(xr,self.z), 
                                        self.R)
                                        
                # Put new landmark in the dictionary of the obsaerved features
                x_f=self.ekf.x_[idx]
                y_f=self.ekf.x_[idx+1]
                self.traffic_signs[self.actual_class_id] = [x_f,y_f,idx]

                self.camera = False
                self.pub=True 
        
        # Publish results
        if self.pub:
            self.publish_results()
            self.pub = False
            self.actual_class_id = None



if __name__ == '__main__':
    x_init = np.array([0.0,0.0,0.0])
    P_init = np.eye(3)

    # ROS initializzation
    rospy.init_node('localization_node')
    localization_node = LocalizationNode(x_init,P_init,"/turtlebot/odom","/preception_topic")

    # Filter at 10 Hz
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        # Iterate filter
        localization_node.iterate()
        r.sleep()