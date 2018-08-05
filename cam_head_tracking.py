#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real time webcam based face tracking pipeline. Capable of tracking faces,
landmarks, and head pose through a webcam.

Created on Fri Jul 13 13:14:57 2018

@author: eric yang
"""
import FaceVar
from kf_2points import kf_2points
from collections import deque
import numpy as np
import glob
from face_utils import box_iou2, draw_box_on_image, draw_marks_on_image, draw_BBox
from scipy.optimize import linear_sum_assignment
import cv2
from FaceDetector import FaceDetector
import time
from MarkDetector import MarkDetector
from MarkStabilizer import MarkStabilizer
from PoseEstimator import PoseEstimator
from kf_FaceTracker import Tracker

def make_resolution(cam, x, y):
    cam.set(3, x)
    cam.set(4, y)

def make_480p(cam):
    cam.set(3, 640)
    cam.set(4, 480)


def main():
    video_src = 0  # 0 means webcam; set file name if video
    cam = cv2.VideoCapture(video_src)
    # make_resolution(cam, 1200, 600)
    _, sample_frame = cam.read()  # get 1 sample frame to setup codes
    height, width = sample_frame.shape[:2]
    face_detector = FaceDetector()
    tracker = Tracker(FaceVar.IOU_THRESHOLD, img_size=(height, width))
    
    while True:
        frame_got, frame = cam.read() # read an image from camera
        
        if frame_got is False: # if not getting an image, just break
            print('Camera is not getting images')
            break
        
        # detect faces from the current image 
        conf, boxes = face_detector.get_faceboxes(image=frame, threshold=0.9)
        
        # associate the current detection with existing tracks
        matches, unmatched_dets, unmatched_tracks = tracker.assign_detections_to_trackers(boxes)
        
        # update the tracking system
        tracker.update(frame, boxes, matches, unmatched_dets, unmatched_tracks)
        
        # annotate curent image (optional for bonding box, landmarks, head pose. all fater kalman filter)
        frame = tracker.annotate_BBox(frame)
        
        # choose if draw orignal detection bonding box on image, without kalman filter.
        if FaceVar.DRAW_ORIG_BBOX:
            frame = draw_BBox(image=frame, faceboxes=boxes, confidences=conf)     
        
        cv2.imshow("preview", frame)
        if cv2.waitKey(10) == 27:
            break
        
if __name__ == '__main__':
    main()