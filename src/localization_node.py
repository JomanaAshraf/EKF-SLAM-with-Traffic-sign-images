#!/usr/bin/env python

import rospy
import math
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from localization_project.msg import perception_data
import numpy as np
import slam_functions as func
from ekf_slam import EKFSlam

class LocalizationNode(EKFSlam):
    def __init__(self, xr_init, Pr_init,odom_topic,detected_sign_topic,initial_map = None):
        super().__init__(xr_init, Pr_init, 100, 2, initial_map)

        # Publishers
        self.pub_odom = rospy.Publisher("predicted_position", Odometry,queue_size=2)
        self.pub_uncertainity = rospy.Publisher("uncertainity", Marker,queue_size=2)
        self.feature_pub= rospy.Publisher("feature_markers", MarkerArray,queue_size=2)

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
        # self.Q = np.array([[0.25**2,0],
        #                     [0.25**2,0]])
        
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

        # Filter
        self.ekf = EKFSlam(xr_init, Pr_init, 100, 3, initial_map)

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
        self.rho = msg.rho_projected
        self.theta=msg.theta
        self.class_id=msg.class_id
        print("self.rho",self.rho)
        print("self.theta",self.theta)
        self.z=np.array([self.rho,self.theta])
        self.xf=np.array([self.ekf.x_[0]+0.138*math.cos(self.ekf.x_[2])+self.rho*math.sin(self.ekf.x_[2])*math.sin(self.theta)+self.rho*math.cos(self.ekf.x_[2])*math.cos(self.theta),
                        self.ekf.x_[1]+0.138*math.sin(self.ekf.x_[2])-self.rho*math.cos(self.ekf.x_[2])*math.sin(self.theta)+self.rho*math.sin(self.ekf.x_[2])*math.cos(self.theta)])

        print("self.xf",self.xf)
        # Flag
        self.camera = True


    def get_ekf_msgs(self):
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
        msg_odom.pose.pose.position.x = self.ekf.x_[0]
        msg_odom.pose.pose.position.y = self.ekf.x_[1]
        msg_odom.pose.pose.position.z = 0
        quat = func.quaternion_from_yaw(self.ekf.x_[2])
        msg_odom.pose.pose.orientation.x = quat[0]
        msg_odom.pose.pose.orientation.y = quat[1]
        msg_odom.pose.pose.orientation.z = quat[2]
        msg_odom.pose.pose.orientation.w = quat[3]
        msg_odom.pose.covariance=[self.ekf.P_[0,0],self.ekf.P_[0,1],0,0,0,self.ekf.P_[0,2],self.ekf.P_[1,0],self.ekf.P_[1,1],0,0,0,self.ekf.P_[1,2]
                            ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,self.ekf.P_[2,0],self.ekf.P_[2,1],0,0,0,self.ekf.P_[2,2]]

        # Uncertainity
        # print("byeeeeeeeeeeeeeeee")
        uncert = self.ekf.P_[0:3, 0:3].copy()
        val, vec = np.linalg.eigh(uncert)
        yaw = np.arctan2(vec[1, 0], vec[0, 0])
        quat = func.quaternion_from_yaw(yaw)
        msg_ellipse = Marker()
        msg_ellipse.header.frame_id = "base_link"
        msg_ellipse.header.stamp = time
        msg_ellipse.type = Marker.CYLINDER
        msg_ellipse.pose.position.x = self.ekf.x_[0]
        msg_ellipse.pose.position.y = self.ekf.x_[1]
        msg_ellipse.pose.position.z = -0.1  # below others
        msg_ellipse.pose.orientation.x = quat[0]
        msg_ellipse.pose.orientation.y = quat[1]
        msg_ellipse.pose.orientation.z = quat[2]
        msg_ellipse.pose.orientation.w = quat[3]
        msg_ellipse.scale.x = math.sqrt(val[0])
        msg_ellipse.scale.y = math.sqrt(val[1])
        msg_ellipse.scale.z = 0.05
        msg_ellipse.color.a = 0.6
        msg_ellipse.color.r = 0.0
        msg_ellipse.color.g = 1
        msg_ellipse.color.b = 0.0

        return msg_odom, msg_ellipse

    
    def publish_results(self):
        """Publishe results from the filter."""

        # Get filter data
        predicted_odom, ellipse = self.get_ekf_msgs()

        # Publish results
        self.pub_odom.publish(predicted_odom)
        self.pub_uncertainity.publish(ellipse)

    def iterate(self):
        """Main loop of the filter."""
        # Prediction
        if self.new_odom:
            check_rotation=math.isclose(self.yaw,self.previous_yaw,abs_tol=0.01)
            # print("checkkkkkkkkkkkkk",check_rotation)
            if self.uk[0]!=0 or self.uk[1]!=0 or check_rotation == False:
                [xr, pr, prm] = self.ekf.prediction(self.uk,self.Q)
                self.ekf.applyPrediction(xr,pr,prm)
                # print("prediction x", self.ekf.x_[0:3])
                self.last_odom = self.odom  # new start odom for incremental
                self.new_odom = False
                self.pub=True
                self.time = self.odomtime
                print("prediction",self.ekf.P_[0:3, 0:3])

        if self.camera:
            # Known landmark! Do Upate
            xr = self.ekf.x_[0:3]
            print(xr)
            ma = MarkerArray()
            if self.class_id in self.traffic_signs.keys():
                idx = self.traffic_signs[self.class_id][2]
                print("updatinggggggggggggggggggg",idx)
                xr = self.ekf.x_[0:3]
                xf = self.xf
                [x, P] = self.ekf.update(self.z,
                                    self.R,
                                    idx,
                                    func.h(xr, xf), 
                                    func.Jhxr(xr, xf),
                                    func.Jhxf(xr, xf),
                                    func.Jhv())
                self.ekf.applyUpdate(x, P)
                self.camera = False 
                self.pub=True 
                print("before update",self.traffic_signs)
                self.traffic_signs[self.class_id][0]=self.ekf.x_[3+(idx*2)]
                self.traffic_signs[self.class_id][1]=self.ekf.x_[3+((idx+1)*2)]
                print("after update",self.traffic_signs)
                print("update",self.ekf.P_[0:3, 0:3])

            #Unknown landmark! Do initialization
            else: 
                idx = self.ekf.add_landmark(func.g(xr,self.z), 
                                        func.Jgxr(xr,self.z), 
                                        func.Jgz(xr,self.z), 
                                        self.R)
                                        
                ## Put new landmark in the dictionary of the obsaerved features
                x_f=self.xf[0]
                y_f=self.xf[1]
                print("x_f",x_f)
                print("y_f",y_f)
                self.traffic_signs[self.class_id] = [x_f,y_f,idx]

                self.camera = False
                self.pub=True 
            print(self.traffic_signs)

            if self.traffic_signs is not None:
                keys=list(self.traffic_signs)
                for idx, id in enumerate(keys): 
                    # print("ekfslaaaaaaam",self.ekf.x_)
                    feature_id=self.traffic_signs[keys[idx]][2]*2
                    print("feature_id",3+feature_id)
                    print("in markers x",self.traffic_signs[keys[idx]][0]) 
                    print("in markers y",self.traffic_signs[keys[idx]][1])      
                    print("state vector x",self.ekf.x_[3]) 
                    print("state vector y",self.ekf.x_[3+feature_id+1])         
                    m = Marker()
                    m.id = id
                    m.header.stamp = rospy.Time.now()
                    m.header.frame_id = "odom"
                    m.type = m.CUBE
                    m.pose.position.z = 0.0
                    m.pose.position.x = self.ekf.x_[3+feature_id]
                    m.pose.position.y = self.ekf.x_[3+feature_id+1]

                    m.pose.orientation.w = 1.0

                    m.color.a = 1.0
                    m.color.g = 1.0

                    m.scale.x = 0.1
                    m.scale.y = 0.1
                    m.scale.z = 0.1

                    ma.markers.append(m)
                    self.feature_pub.publish(ma) 
        # Publish results
        if self.pub:
            self.publish_results()
            self.pub = False



if __name__ == '__main__':
    x_init = np.array([0.0,0.0,0.0])
    P_init = np.eye(3) * 0.001

    # ROS initializzation
    rospy.init_node('localization_node')
    localization_node = LocalizationNode(x_init,P_init,"/turtlebot/odom","/preception_topic")

    # Filter at 10 Hz
    r = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        # Iterate filter
        localization_node.iterate()
        r.sleep()