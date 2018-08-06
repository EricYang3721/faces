#========== Global Tracking vars ==============
'''The variables for Face, landmakr, head pose tracking system.'''
# Face bonding box measurement covariance for kalman filter kf_2points
COV_MEASUREMENT=0.001

# Face Bonding box process covariance for kalman filter kf_2points
COV_PROCESS=0.1

# Max Trace Length
MAX_TRACE_LENGTH=20

# no. of consecutive matches needed to draw tracking
MIN_HITS = 10

# no.of consecutive unmatched detection before a track is deleted
MAX_AGE = 45

# IOU threshold for detection-tracks assignement
IOU_THRESHOLD = 0.3

# Number of frames to trace the face bonding boxes
FACEBOX_TRACE_LENGTH = 20

# Whether draw face detection box after kalman filter
DRAW_DETECTION_BOX = True

# Whether draw original bonding box from detector
DRAW_ORIG_BBOX = True 

#===============Landmakr tracking parameter=======================
# Turn on landmark tracking or not: true--on, false--off
LADNMARK_ON = True

# Face landmark measurement covariance for kalman filter kf_2points
MARK_COV_MEASUREMENT=0.1

# Face landmark process covariance for kalman filter kf_2points
MARK_COV_PROCESS=0.01

# Landmarks tracing length
LANDMARK_TRACE_LENGTH=20



#=============== Head Pose tracking parameter=================

# Turn on Head Pose tracking or not: true--on, false--off
HEADPOSE_ON = True

# Face landmark measurement covariance for kalman filter kf_2points
POSE_COV_MEASUREMENT=0.1

# Face landmark process covariance for kalman filter kf_2points
POSE_COV_PROCESS=0.01

# Landmarks tracing length
POSE_TRACE_LENGTH=20
