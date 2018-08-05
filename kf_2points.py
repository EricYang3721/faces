# kalman filter with 2 points together
import numpy as np
import cv2

''' Tracker with 2 2D points at the same time. This is used to kalman filter the 
face bonding boxes between frame (or the arrow indicating head pose). It is easier to 
use compare to use 2 individual kalman filters for each point on the box.'''


class kf_2points:
    '''Kalman filter with 2 2D ponits (x, y) on an image'''
    def __init__(self, initial_state, cov_process = 0.001, cov_measure=0.01):
        '''Initialize the kalman filter.
        Input: initial_state --- the state to initialize the kalman filter. 
                                a list of 4 elements[x1, y1, x2, y2]
            cov_process --- process covariance
            cov_meansure --- measurement covariance'''
        # input dimension
        self.input_dim = 4
        
        # Set up process and input dimensions
        self.state_num = 2*self.input_dim
        self.measure_num = self.input_dim

        # initiate filter from opencv
        # No control parameter for now        
        self.filter = cv2.KalmanFilter(self.state_num,
                                       self.measure_num,
                                       0)
        
        # Store the state
        self.state = np.array([[np.float32(initial_state[0])],
                                     [np.float32(initial_state[1])],
                                     [np.float32(initial_state[2])],
                                     [np.float32(initial_state[3])],
                                     [0], [0], [0], [0]])
        #self.state = np.zeros((self.state_num, 1), np.float32)
        
        # store the measurement results
        self.measurement = np.zeros((self.measure_num, 1), np.float32)
        
        # Store the prediction 
        self.prediction = np.zeros((self.state_num, 1), np.float32)
        
        # set up the parameters for the kalmanfilter
        self.filter.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],
                                                 [0,1,0,0,0,1,0,0],
                                                 [0,0,1,0,0,0,1,0],
                                                 [0,0,0,1,0,0,0,1],
                                                 [0,0,0,0,1,0,0,0],
                                                 [0,0,0,0,0,1,0,0],
                                                 [0,0,0,0,0,0,1,0],
                                                 [0,0,0,0,0,0,0,1]], np.float32)
        self.filter.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],
                                                  [0,1,0,0,0,0,0,0],
                                                  [0,0,1,0,0,0,0,0],
                                                  [0,0,0,1,0,0,0,0]], np.float32)
        self.filter.processNoiseCov = np.eye(self.state_num, dtype=np.float32)*cov_process

        self.filter.measurementNoiseCov = np.eye(self.measure_num, dtype=np.float32)*cov_measure
        
    def predict(self):
        # make predictions with previous results, and return prediction 
        self.prediction = self.filter.predict()
        return self.prediction
    
    def correct(self, measurement):
        # correct the prediction with new measurements, and return the new state
        self.measurement = np.array([[np.float32(measurement[0])],
                                     [np.float32(measurement[1])],
                                     [np.float32(measurement[2])],
                                     [np.float32(measurement[3])]])
                # correct according to measurement
        self.filter.correct(self.measurement)
        # update the state value
        self.state = self.filter.statePost
        return self.state
    
    def get_results(self):
        # get the new corrected state, and arrange them in a list
        return [(int)(self.state[0]), (int)(self.state[1]), (int)(self.state[2]), (int)(self.state[3])]
    



def main():
    """Test code"""
    global mp
    mp = np.array((1, 2, 3, 4), np.float32)  # measurement

    def onmouse(k, x, y, s, p):
        global mp
        mp = np.array([[np.float32(x)], [np.float32(y)], [np.float32(x+10.)], [np.float32(y+10.)]])

    cv2.namedWindow("kalman")
    cv2.setMouseCallback("kalman", onmouse)
    kalman = kf_2points([0,0,0,0], 1, 1)
    frame = np.zeros((480, 640, 3), np.uint8)  # drawing canvas

    while True:
        kalman.predict()
        kalman.correct(mp)
        #point = kalman.prediction
        #state = kalman.filter.statePost
        result = kalman.get_results()
        cv2.circle(frame, (result[0], result[1]), 2, (255, 0, 0), -1)
        cv2.circle(frame, (result[2], result[3]), 2, (255, 0, 0), -1)
        cv2.circle(frame, (mp[0], mp[1]), 2, (0, 255, 0), -1)
        cv2.circle(frame, (mp[2], mp[3]), 2, (0, 255, 0), -1)
        cv2.imshow("kalman", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    main()
            