import re
import numpy as np
import math

from tf.transformations import euler_from_quaternion, quaternion_from_euler

def angle_wrap(ang):
    """
    Return the angle normalized between [-pi, pi].

    Works with numbers and numpy arrays.

    :param ang: the input angle/s.
    :type ang: float, numpy.ndarray
    :returns: angle normalized between [-pi, pi].
    :rtype: float, numpy.ndarray
    """
    ang = ang % (2 * np.pi)
    if (isinstance(ang, int) or isinstance(ang, float)) and (ang > np.pi):
        ang -= 2 * np.pi
    elif isinstance(ang, np.ndarray):
        ang[ang > np.pi] -= 2 * np.pi
    return ang

def comp(a, b):
    """
    Compose matrices a and b.

    b is the matrix that has to be transformed into a space. Usually used to
    add the vehicle odometry

    b = [x' y' theta'] in the vehicle frame, to the vehicle position
    a = [x y theta] in the world frame, returning world frame coordinates.

    :param numpy.ndarray a: [x y theta] in the world frame
    :param numpy.ndarray b: [x y theta] in the vehicle frame
    :returns: the composed matrix a+b
    :rtype: numpy.ndarray
    """
    c1 = math.cos(a[2]) * b[0] - math.sin(a[2]) * b[0] + a[0]
    c2 = math.sin(a[2]) * b[0] + math.cos(a[2]) * b[0] + a[1]
    c3 = a[2] + b[2]
    c3 = angle_wrap(c3)
    C = np.array([c1, c2, c3])
    return C

def yaw_from_quaternion(quat):
    """
    Extract yaw from a geometry_msgs.msg.Quaternion.

    :param geometry_msgs.msg.Quaternion quat: the quaternion.
    :returns: yaw angle from the quaternion.
    :rtype: :py:obj:`float`
    """
    return euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

def quaternion_from_yaw(yaw):
    """
    Create a Quaternion from a yaw angle.

    :param float yawt: the yaw angle.
    :returns: the quaternion.
    :rtype: :py:obj:`tuple`
    """
    return quaternion_from_euler(0.0, 0.0, yaw)

# The Differential Drive Prediction equations
def calculate_f(xk,uk) -> np.ndarray:
    return comp(xk,uk)

def calculate_Jfx(xk, uk) -> np.ndarray:
    dt = 1
    return np.array([[dt, 0, -(uk[0]*math.sin(xk[2]))-(uk[1]*math.cos(xk[2]))], 
                    [0, dt, uk[0]*math.cos(xk[2])-(uk[1]*math.sin(xk[2]))], 
                    [0, 0,  dt ]])

def calculate_Jfw(xk,dt=None) -> np.ndarray:
    # dt=0.002
    return np.array([[math.cos(xk[2]),-math.sin(xk[2]), 0], 
                         [math.sin(xk[2]),math.cos(xk[2]), 0],
                         [0,0,1]])

# def calculate_Jfw(xk,dt=None) -> np.ndarray:
#     dt=0.002
#     return np.array([[math.cos(xk[2])*dt, 0], 
#                          [math.sin(xk[2])*dt, 0],
#                          [0,dt]])

def h(xr,xf):  
        '''
        Compute the expected measurement z = h(xr, xl, v), where xr is the robot state [x,y,\theta] and xl the pinger state [x,y]
        Input:
        - xr: numpy array of shape (3,) representing the robot state [x,y,\theta]
        - xf: numpy array of shape (2,) representing the [xf,yf] position of the detected traffic sign in the world
        Return: numpy array of shape (2,) representing the expected measurement [x,y]
        '''
        expected_x = xf[0]*math.sin(xr[2])-xf[1]*math.cos(xr[2])-xr[0]*math.sin(xr[2])+xr[1]*math.cos(xr[2])

        expected_z = xf[0]*math.cos(xr[2])+xf[1]*math.sin(xr[2])-xr[0]*math.cos(xr[2])-xr[1]*math.sin(xr[2])-0.138

        return np.array([expected_x,expected_z])

def Jhxr(xr, xf) -> np.ndarray: # method "1"
    ''' 
    Compute the Jacobian of h(xr, xl ,v) with respect to xr, at point (xr, xl)
    Input:
    - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
    - xf: numpy array of shape (2,) representing the [xf,yf] position of the detected traffic sign in the world
    return: numpy matrix of shape (2, 3) (The Jacobian)
    '''


    # jhxr_d=np.array([-(xl[0]-xr[0])/d,-(xl[1]-xr[1])/d,0]) 
    # jhxr_r=np.array([((xl[1]-xr[1])/((xl[1]-xr[1])**2+(xl[0]-xr[0])**2)),(-(xl[0]-xr[0])/((xl[1]-xr[1])**2+(xl[0]-xr[0])**2)),-1])
    # x=np.array([jhxr_d,jhxr_r])
    # print("xxxxxxxxxxxxx",x.shape)
    # return np.array([jhxr_d,jhxr_r])

    jhxr_x=np.array([-math.sin(xr[2]), math.cos(xr[2]), xf[0]*math.cos(xr[2])+xf[1]*math.sin(xr[2])
                    -xr[0]*math.cos(xr[2])-xr[1]*math.sin(xr[2])])


    jhxr_y=np.array([-math.cos(xr[2]), -math.sin(xr[2]), -xf[0]*math.sin(xr[2])+xf[1]*math.cos(xr[2])
                    +xr[0]*math.sin(xr[2])-xr[1]*math.cos(xr[2])])
    x=np.array([jhxr_x,jhxr_y])
    # print("xxxxxxxxxx",x)
    return np.array([jhxr_x,jhxr_y])

def Jhxf(xr, xf) -> np.ndarray:  # method "1"
    '''
    Compute the Jacobian of h(xr, xl ,v) with respect to xl, at point (xr, xl)
    Input:
    - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
    - xf: numpy array of shape (2,) representing the [xf,yf] position of the detected traffic sign in the world
    return: numpy matrix of shape (2, 3) (The Jacobian)
    '''

    jhxl_x=np.array([math.sin(xr[2]), -math.cos(xr[2])])

    jhxl_y=np.array([math.cos(xr[2]),math.sin(xr[2])])

    return np.array([jhxl_x, jhxl_y])

def Jhv(xr = None, xf= None) -> np.ndarray:
    # Noise is assumed independent (z = h(x) + v)
    return np.eye(2)


def g(xr,z):
    '''
    Compute the inverse observation
    Input:
    - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
    - z: measurement which is [rho_projected,angle]
    '''

    xl = xr[0]+0.138*math.cos(xr[2])+z[0]*math.sin(xr[2])*math.sin(z[1])+z[0]*math.cos(xr[2])*math.cos(z[1])

    yl = xr[1]+0.138*math.sin(xr[2])-z[0]*math.cos(xr[2])*math.sin(z[1])+z[0]*math.sin(xr[2])*math.cos(z[1])

    return np.array([xl,yl])

def Jgxr(xr,z) -> np.ndarray:

    Jgxr_1=np.array([1, 0, -0.138*math.sin(xr[2])+z[0]*math.cos(xr[2])*math.sin(z[1])-z[0]*math.sin(xr[2])*math.cos(z[1])])

    Jgxr_2=np.array([ 0, 1, 0.138*math.cos(xr[2])+z[0]*math.sin(xr[2])*math.sin(z[1])+z[0]*math.cos(xr[2])*math.cos(z[1])])

    return np.array([Jgxr_1,Jgxr_2])

def Jgz(xr,z) -> np.ndarray:

    Jgz_1=np.array([math.sin(xr[2])*math.sin(z[1])+math.cos(xr[2])*math.cos(z[1]),
                    z[0]*math.sin(xr[2])*math.cos(z[1])-z[0]*math.cos(xr[2])*math.sin(z[1])])

    Jgz_2=np.array([math.sin(xr[2])*math.cos(z[1])-math.cos(xr[2])*math.sin(z[1]),  
                    -z[0]*math.cos(xr[2])*math.cos(z[1])-z[0]*math.sin(xr[2])*math.sin(z[1])])

    return np.array([Jgz_1,Jgz_2])

def data_association(ground_truth,xr,p_robot,z,R):
    Dmin = math.inf
    nearest = -1 
    ground_truth_list=list(ground_truth)
    for j in range (len(ground_truth)):
        Jhxj = Jhxr(xr,ground_truth[ground_truth_list[j]])#This variable is the jacobian of h() in the current robot believe and map feature j
        hfj = h(xr, ground_truth[ground_truth_list[j]]) #The expected measurement if the associated map feature is j
        # print("jhxj",Jhxj.shape)
        Pfj = Jhxj@p_robot@np.transpose(Jhxj) #The expected measurement uncertainty if the associated map feature is j

        #Mahalanobis
        v=z-hfj
        S=R+Pfj
        Dij =v@np.linalg.inv(S)@v.T  # This variable is the mahalanobis distance
        
        if Dij < Dmin:
            nearest = j
            Dmin = Dij 
    
    hyp = nearest

    return hyp

