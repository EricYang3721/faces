# Multi-face tracking application

This is an implementation of a multi-face tracking system using kalman filter to track multiple faces in video/webcam, including faces and landmarks detection, head pose estimation.  

### 1. Structure of the implementation

The implementation is organized as following

![](https://github.com/EricYang3721/faces/blob/master/photo/code_structure.jpg)

FaceDetector.py adopted an Single Shot Detector (SSD) for identifying faces in given images. Details about the SSD is could be found SSD: Single Shot MultiBox Detector https://arxiv.org/abs/1512.02325 

MarkDetector.py uses a CNN to identify the landmark locations on given faces. It firstly crops faces from input image by expanding the bonding boxes from the FaceDetector. Then, these cropped faces are sent to CNN to identify the facial landmarks. Finally, these landmark locations are fitted back into the coordinates of the  original image. 

PoseEstimator.py estimates the head pose by mapping the detected 2D facial landmarks from MarkDetector to the average 3D facial landmarks. Two ways to estimate head pose are implemented, one uses 6 landmarks, and the other uses all 68 landmarks detected. 

kf_2points.py is a Kalman filter for bonding boxes. Each bonding box has 2 2D points, and these 2 points kalman filters simultaneously with kf_2points.py

MarkStablizer.py is is a Kalman filter implemetation for 1D and 2D points, which are used for landmarks (2D points), and head pose (considering each entry in rotation vector and translation vector in 3D to 2D mapping as a 1D point). 

cam_head_tracking.py is the application which integrates all above functions for real time face tracking & analysis based on a webcam. 

### 2. To run the app  

This implementation depends on Python 3.6, OpenCV 3.2.0, tensorflow 1.8.0, numpy. The codes are compile in Ubuntu 16.04 LTS.

All source code are located in the folder of ./faces/. 

The ssd face detection model (res10_300x300_ssd_iter_140000.caffemodel and res10_300x300_ssd_iter_140000.prototxt) should be put in the folder of ./faces/models/face_detector. The model could be download from 

[1]: https://github.com/thegopieffect/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel
[2]: https://github.com/thegopieffect/computer_vision/blob/master/CAFFE_DNN/deploy.prototxt.txt

The landmark detector model (frozen_inference_graph.pb) should be saved in the folder of  faces/models/landmark_detector. The model for the landmark detector could be downloaded from 

[3]: https://github.com/yinguobing/head-pose-estimation/blob/master/assets/frozen_inference_graph.pb

The 68 3D facial landmarks for an average face in model_landmark.txt should be save in the folder of /faces/models/

To run the code, just run following code:

```python
python cam_head_tracking.py
```

The detection/estimation of faces, landmarks and head pose could be independently turn on or off in FaceVar.py by setting the true of false with following constants.  

```python
DRAW_ORIG_BBOX = True   # drawn the original bonding box from the FaceDetector()
DRAW_DETECTION_BOX = True  # drawn tracked face bonding box
LADNMARK_ON = True   # turn on landmark tracking
HEADPOSE_ON = True   # turn on head pose estimation
```

Other parameters controlling the tracking system could also be adjusted inside the FaceVar.py. Please refer to the annotation in the file for more details.

### 3. Some results

![](https://github.com/EricYang3721/faces/blob/master/photo/results.jpg)

Once running the cam_head_tracking.py, the above image would pop out. 

The yellow rectangle is the original bonding box directly detection, the number at the bottom right corner is the confidence of this detection. 

The red box is the face bonding box after kalman filter, and the number on the top left corner is the id of the face identified in current streaming. 

The green dots are facial landmarks, and the blue wedge is the head pose direction, both are results after kalman filter. 

All detections/estimation on face, landmarks and head pose could independently turned on or off as explained in Section 2. 

### 4. Other functions

A few functions could be used in adjusting the parameters in FaceVar.py

1. Face re-identification is also implemented in case of occlusions. This re-identification relies on the IOU score. If a track loses its detection less than certain number of frames, it could be re-identified as the same face. This number of frames is defined 

   ```python
   MAX_AGE = 45   # no.of consecutive unmatched detection before a track is deleted
   ```

2. The number of images to draw a tracking is defined as

   ```python
   MIN_HITS = 10   # no. of consecutive matches needed to draw tracking
   ```


### 5. References

This implementation referred or adapted partially from following sources:

[1]: https://github.com/yinguobing/head-pose-estimation
[2]: https://yinguobing.com/facial-landmark-localization-by-deep-learning-network-model/
[3]: https://github.com/kcg2015/Vehicle-Detection-and-Tracking/blob/master/helpers.py
[4]: https://github.com/mabhisharma/Multi-Object-Tracking-with-Kalman-Filter
[5]: https://github.com/kcg2015/Vehicle-Detection-and-Tracking

