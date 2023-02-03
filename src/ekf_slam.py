import numpy as np
import slam_functions as func

class EKFSlam:
    '''
    Class to encapsulate the EKFSlam Problem.
    It stores:
        - x_: np.array representing the robot and map state x = [xr | xl1 | xl2 .... xln], where xr = [x,y,\theta] of the robot and xl = [x,y] of a landmark
        - P_: np matrix representing the robot and map covariance matrix
        - n_landmarks_ : int representing the number of landmarks in the map
        - landmark_size_ : int representing the variables that represent a landmark (e.g., for pingers its 2: [x,y])

    Note that all internal variables finish with an underscore. This is done in order to differenciate it from the class methods (and its usually standard to represent object variables)
    
    Since x_ and P_ will increase when new landmarks are detected, x_ and P_ matrices are pre-allocated for a maximum number of landmarks.
    Therefore, special care has to be taken in order to use or write to the right sections of x_ and P_ for operations, and not overwrite them with numpy arrays
    For this reason, a set of functions are built so we can access and write this data wisely
    A set of "getters" (functions that return information of the class) are defined to access data. They will return numpy views, which allow 
    to define subvectors and submatrices of x_ and P_ by reference (so if we modify them, it will modify the parent)
    A set of "setters" (functions to write information to the class) are defined to modify x_ and P_. The idea is to do it only here, so we don't do mistakes from outside the class
    '''
    def __init__(self, x_robot_init, P_robot_init, max_landmarks = 100, landmark_size = 2, map = None) -> None:
        max_x_size = 3 + max_landmarks * landmark_size
        self.x_ = np.zeros(max_x_size,)
        self.P_ = np.zeros((max_x_size, max_x_size))

        self.x_[0:3] = x_robot_init
        self.P_[0:3, 0:3] = P_robot_init

        self.n_landmarks_ = 0
        self.landmark_size_ = landmark_size

    #Getters
    ##Return views to self.x_ and self.P_ for efficiency and readibility!

    def x(self):
        '''
        return a numpy view of shape (3 + n * s,) where n is the number of landmarks and s is the landmark size
        '''
        return self.x_[0:3 + self.n_landmarks_ * self.landmark_size_]
    
    def P(self):
        '''
        return a numpy view of shape (3 + n*s, 3 + n*s) representing the robot state covariance
        '''
        return self.P_[0:3 + self.n_landmarks_ * self.landmark_size_, 0:3 + self.n_landmarks_ * self.landmark_size_]

    def xr(self):
        '''
        return a numpy view of shape (3,) representing the robot state [x,y,\theta]
        '''
        return self.x_[0:3]

    def Prr(self):
        '''
        return a numpy view of shape (3,3) representing the robot state covariance
        '''
        return self.P_[0:3, 0:3]

    def xl(self, idx):
        '''
        return a numpy view of shape (s,) representing a landmark state (s is 2 for a pinger [x,y])
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        if idx >= self.n_landmarks_:
            return None
        f = 3 + idx * self.landmark_size_
        t=f+2
        # t = 3 + (idx + 1) * self.landmark_size_
        return self.x_[f:t]

    def Prm(self):
        '''
        return a numpy view of shape (3, n * s) representing the robot-map correlation matrix where n is the number of landmarks and s is the landmark size
        '''
        return self.P_[0:3,3:3+self.n_landmarks_*self.landmark_size_]

    def Pmr(self):
        '''
        return a numpy view of shape (n * s, 3) representing the map-robot correlation matrix where n is the number of landmarks and s is the landmark size
        '''
        return self.P_[3:3+self.n_landmarks_*self.landmark_size_,0:3]

    def Pmm(self):
        '''
        return a numpy view of shape (n * s, n * s) representing the map covariance matrix where n is the number of landmarks and s is the landmark size
        '''
        return self.P_[3:3+self.n_landmarks_*self.landmark_size_,3:3+self.n_landmarks_*self.landmark_size_]

    def Prl(self, idx):
        '''
        return a numpy view of shape (3, s) representing a robot-landmark correlation matrix where s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        if idx >= self.n_landmarks_:
            return None
        f = 3 + idx * self.landmark_size_
        t=f+2
        # t = 3 + (idx + 1) * self.landmark_size_
        return self.P_[0:3,f:t]

    def Plr(self, idx):
        '''
        return a numpy view of shape (s, 3) representing a landmark-robot correlation matrix where s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        if idx >= self.n_landmarks_:
            return None
        f = 3 + idx * self.landmark_size_
        t=f+2
        # t = 3 + (idx + 1) * self.landmark_size_
        return self.P_[f:t,0:3]

    def Plm(self, idx):
        '''
        return a numpy view of shape (s, n * s) representing a landmark-mark correlation matrix where n is the number of landmarks and s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        if idx >= self.n_landmarks_:
            return None
        f = 3 + idx * self.landmark_size_
        t=f+2
        # t = 3 + (idx + 1) * self.landmark_size_
        return self.P_[f:t,3:3+self.n_landmarks_*self.landmark_size_]

    def Pml(self, idx):
        '''
        return a numpy view of shape (n * s, s) representing a map-landmark correlation matrix where n is the number of landmarks and s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        if idx >= self.n_landmarks_:
            return None
        f = 3 + idx * self.landmark_size_
        t=f+2
        # t = 3 + (idx + 1) * self.landmark_size_
        return self.P_[3:3+self.n_landmarks_*self.landmark_size_,f:t]


    def Pll(self, idx):
        '''
        return a numpy view of shape (s,s) representing the landmark covariance matrix where s is the size of the landmark
        Input:
            -idx : an integer representing the landmark position in the map
        '''
        if idx >= self.n_landmarks_:
            return None
        f = 3 + idx * self.landmark_size_
        t=f+2
        # t = 3 + (idx + 1) * self.landmark_size_
        return self.P_[f:t,f:t]
    
    def prediction(self, u, Q): 
        '''
        Compute the prediction step of the EKF.
        Input: 
            - u : np array representing the control signal
            - Q : np matrix representing the noise of the control signal
            - dt: float representing the increment of time
        Return:
            - xr: np array of shape (3,) representing the predicted robot state
            - Prr: np matrix of shape (3,3) representing the predicted robot covariance
            - Prm: np matrix of shape (3, n * s) representing the predicted robot-map correlation matrix
        '''
        assert isinstance(u, np.ndarray)
        assert isinstance(Q, np.ndarray)
        
        xr = func.calculate_f(self.x_[0:3],u)
        A = func.calculate_Jfx(self.x_[0:3],u)
        W = func.calculate_Jfw(self.x_[0:3])
    
        Prr =A@self.Prr()@np.transpose(A)+W@Q@np.transpose(W)
        Prm = A@self.Prm()
        
        return [xr, Prr, Prm] 
        
    # Setters:  Functions to apply changes!
    def applyPrediction(self, xr, Prr, Prm):
        '''
        Apply a prediction (Be careful fill self.x_ and self.P_, but don't overwrite it!)
        Input: 
            - xr: np array of shape (3,) representing the predicted robot state
            - Prr: np matrix of shape (3,3) representing the predicted robot covariance
            - Prm: np matrix of shape (3, n * s) representing the predicted robot-map correlation matrix
        '''
        self.x_[0:3] = xr
        self.P_[0:3, 0:3] = Prr
        self.P_[0:3, 3: 3 + self.n_landmarks_*self.landmark_size_] = Prm
        self.P_[3: 3 + self.n_landmarks_*self.landmark_size_, 0:3] = Prm.transpose()

    def update(self, z, R, lid, hx, Hr, Hl, Hv):
        '''
        Compute the update step of the EKF.
        Input: 
            - z : np array representing the measurement
            - R : np matrix representing the noise of the measurement
            - lid: int representing the landmark id of the measurement
            - hx: np array representing the expected measurement
            - Hr: np matrix representing the jacobian of hx with respect to xr
            - Hl: np matrix representing the jacobian of hx with respect to xl
            - Hv: np matrix representing the jacobian of hx with respect to v
        Return:
            - x: np array of shape (3 + n * s,) representing the updated robot and map state
            - P: np matrix of shape (3 + n*s, 3 + n*s) representing the updated robot and map covariance
        '''
        assert isinstance(z, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert isinstance(hx, np.ndarray)
        assert isinstance(Hr, np.ndarray)
        assert isinstance(Hl, np.ndarray)
        assert isinstance(Hv, np.ndarray)
        assert isinstance(lid, int)
 
        # Compute innovation
        y = z - hx
        # for i in range(1, len(y), 2): # Assuming all measurements are [distance, angle]
        #     y[i] = func.angle_wrap(y[i])
        
        H=np.hstack((Hr,Hl))
        p_cov=np.block([[self.Prr(),self.Prl(lid)],[self.Plr(lid),self.Pll(lid)]])
        Z=H@p_cov@np.transpose(H)+R
        k_cov=np.block([[self.Prr(),self.Prl(lid)],[self.Pmr(),self.Pml(lid)]])
        k=k_cov@np.transpose(H)@np.linalg.inv(Z)
        
        x = self.x()+k@y
        P = self.P()-k@Z@np.transpose(k)
        
        return [x, P]
    
    def applyUpdate(self, x, P):
        '''
        Apply a update (Be careful fill self.x_ and self.P_, but don't overwrite it!)
        Input: 
            - x: np array of shape (3 + n*s,) representing the updated state
            - P: np matrix of shape (3 + n*s, 3 + n*s) representing the updated state covariance
        '''
        self.x_[0:3 + self.n_landmarks_ * self.landmark_size_] = x
        self.P_[0:3 + self.n_landmarks_ * self.landmark_size_, 0:3 + self.n_landmarks_ * self.landmark_size_] = P

    def add_landmark(self, xl, Gr, Gz, R):
        '''
        Add a new landmark to the state!
        Input: 
            - xl: np array representing the landmark state 
            - Gr: np matrix representing the jacobian of g(xr, z) with respect to xr
            - Gz: np matrix representing the jacobian of g(xr, z) with respect to z 
        Return:
            - idx: int representing the landmark position in the map
        '''
        idx = self.n_landmarks_
        f = 3+ idx * self.landmark_size_
        t = f+2#3 + (idx + 1) * self.landmark_size_

        ##Add landmark to the state vector (xl)
        self.x_[f:t] = xl
        ##Add landmark uncertainty (Pll)
        self.P_[f:t,f:t] = Gr@self.Prr()@np.transpose(Gr)+Gz@R@np.transpose(Gz)
        ## Fill the cross-variance of the new landmark with the rest of the state (Plx)
        self.P_[f:t,0:3+self.n_landmarks_*self.landmark_size_]=Gr@np.hstack((self.Prr(),self.Prm()))
        ## Fill the cross-variance of the rest of the state with the new landmark (Pxl)
        self.P_[0:3+self.n_landmarks_*self.landmark_size_,f:t] =np.transpose(self.P_[f:t,0:3+self.n_landmarks_*self.landmark_size_])
        
        self.n_landmarks_ += 1 # Update the counter of landmarks
        return idx # Return the position of the landmark in our state