#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eric
"""

''' Tracker with 2 points on an image (bonding box of face detection or 
2 points for head pose (IOU)'''
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


class Track():
    ''' Track class for each bonding box'''
    def __init__(self, detection, trackId, landmarks=0, headpose=0):
        self.KF = kf_2points(initial_state=detection,
                             cov_process=FaceVar.COV_PROCESS,
                             cov_measure=FaceVar.COV_MEASUREMENT)        
        self.KF.predict()        
        self.KF.correct(detection)
        self.trace = deque(maxlen=FaceVar.MAX_TRACE_LENGTH)
        self.prediction = np.array(detection).reshape(1,4)
        self.trackId = trackId 
        self.box = detection
        self.hits = 0 # number of detection matches
        self.num_loss=0 # number of unmatched tracks (track loss)
        self.trace=deque(maxlen=FaceVar.FACEBOX_TRACE_LENGTH) # track the trace for center of the box
        
#        if(FaceVar.LADNMARK_ON):
#            
#            self.initialize_landmarks(landmarks)
#        if(FaceVar.HEADPOSE_ON):
#            self.initialize_headPose(headpose)
        
    def predict(self):
        '''Predict the face bonding box with the kalman filter implemented with 2 points
        on an image plane'''
        self.prediction = np.array(self.KF.predict()).reshape(1,-1)
        self.box = self.KF.get_results()
    def correct(self, detection):
        '''Correct the kkalman filter results with measurements from measurements'''
        self.KF.correct(detection)
        self.box = self.KF.get_results()
    def initialize_landmarks(self, single_face_landmarks68):
        # landmarks68 is 68X2 list         
        ########### initialization could be changed to first detection results
        self.marks_KFs=[]
        for mark in single_face_landmarks68:
            kf = MarkStabilizer(initial_state=[mark[0],mark[1],0,0],
                                             input_dim=2,
                                             cov_process=FaceVar.MARK_COV_PROCESS,
                                             cov_measure=FaceVar.MARK_COV_MEASUREMENT)
            self.marks_KFs.append(kf)
        '''self.marks_KFs = [MarkStabilizer(initial_state=[0,0,0,0],
                                             input_dim=2,
                                             cov_process=FaceVar.MARK_COV_PROCESS,
                                             cov_measure=FaceVar.MARK_COV_MEASUREMENT) for _ in range(68)]
    '''
        self.marks_state = single_face_landmarks68       
        self.marks_trace = deque(maxlen=FaceVar.LANDMARK_TRACE_LENGTH)
    
    def initialize_headPose(self, pose):
        # pose/pose_state are 1X6 array which records the rotation vecotr (frist 3) and translation vector(last 3)
        
        
        '''self.pose_KFs = [MarkStabilizer(initial_state=[pose,0],
                                             input_dim=1,
                                             cov_process=FaceVar.POSE_COV_PROCESS,
                                             cov_measure=FaceVar.POSE_COV_MEASUREMENT) for _ in range(6)]
    '''    
        self.pose_KFs=[]
        for vect in pose:
            #print(vect)
            kf = MarkStabilizer(initial_state=[vect,0],
                                             input_dim=1,
                                             cov_process=FaceVar.POSE_COV_PROCESS,
                                             cov_measure=FaceVar.POSE_COV_MEASUREMENT)
            self.pose_KFs.append(kf)
        self.pose_state = pose      
        self.pose_trace = deque(maxlen=FaceVar.POSE_TRACE_LENGTH)
#        self.predict_headPose()
#        self.correct_headPose(pose)
    
    def predict_marks(self):
        stabilized_marks68 = []
        for stb in self.marks_KFs:
            stb.predict()
            stabilized_marks68.append(stb.get_results())
            #stabilized_marks.append(mark)
        #stabilized_marks68 = np.array([stabilized_marks68])
        self.marks_state = stabilized_marks68

        
    def correct_marks(self, single_face_marks68):
        stabilized_marks68 = []
        for mark, stb in zip(single_face_marks68, self.marks_KFs):
            stb.correct(mark)
            stabilized_marks68.append(stb.get_results())
            #stabilized_marks.append(mark)
        #stabilized_marks68 = np.array([stabilized_marks68])
        self.marks_state = stabilized_marks68
        
        
    def predict_headPose(self):
        stabilized_pose_vector =[]
        for stb in self.pose_KFs:
                stb.predict()
                stabilized_pose_vector.append(stb.get_results())
        self.pose_state = stabilized_pose_vector
        
        
    def correct_headPose(self, pose_vect):
        '''6 scaler as the rotation and translation vector'''
        stabilized_pose_vector =[]
        for vect, stb in zip(pose_vect, self.pose_KFs):
            stb.correct(vect)
            stabilized_pose_vector.append(stb.get_results())  
        self.pose_state = stabilized_pose_vector
        
        
    
class Tracker():
    ''' Trackers for multiple bondign boxes in a video or image'''
    def __init__(self, IOU_threshold, img_size=(720, 1280)):
        '''Initialize the trackers for multiple tracks'''
        self.IOU_threshold = IOU_threshold
        #self.max_frame_skipped = max_frame_skipped
        #self.max_trace_length = max_trace_length
        self.IdCount=0
        self.tracks=[]
        self.markDetector = MarkDetector()
        self.poseDetector = PoseEstimator(img_size=img_size)
    
    def assign_detections_to_trackers(self, detections):
        ''' from existing trackers and new detections, putput mathced detections,
        unmatched trackers, and unmatched detections.
        '''
        IOU_mat = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t,trk in enumerate(self.tracks):
            for d, det in enumerate(detections):
                IOU_mat[t,d] = box_iou2(trk.box, det)       
        # find the matches of tracks and detections, using Hungarian algorithm
        assign_row_ind, assign_col_ind = linear_sum_assignment(-IOU_mat)
        
        unmatched_tracks=[] 
        unmatched_dets=[]
        
        for t, trk in enumerate(self.tracks):
            if t not in assign_row_ind:
                unmatched_tracks.append(t)
        for d, det in enumerate(detections):
            if d not in assign_col_ind:
                unmatched_dets.append(d)
        
        matches = []
        for r, c in zip(assign_row_ind, assign_col_ind):
            if IOU_mat[r][c] < self.IOU_threshold:
                if r not in unmatched_tracks:
                    unmatched_tracks.append(r)
                if c not in unmatched_dets:
                    unmatched_dets.append(c)
            else:
                matches.append([r,c])
        
        if len(matches)==0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.array(matches)
        
        return matches, np.array(unmatched_dets), unmatched_tracks
    
    def update(self, image, detections, matches, unmatched_dets, unmatched_tracks):
        # updating the results with matched and non-matched results
        
        
        # Deal with matched detections
        if len(matches) > 0:
            for trk_idx, det_idx in matches:
                # Face bonding box
                det = detections[det_idx]
                tmp_trk = self.tracks[trk_idx]
                tmp_trk.predict()
                tmp_trk.correct(det)
                tmp_trk.hits +=1
                tmp_trk.trace.append([0.5*(tmp_trk.box[0]+tmp_trk.box[2]),
                                      0.5*(tmp_trk.box[1]+tmp_trk.box[3])])
                if FaceVar.LADNMARK_ON or FaceVar.HEADPOSE_ON:            
                    # Face landmarks
                    single_square_box = self.markDetector.square_single_facebox(tmp_trk.box)
                    single_face_img = self.markDetector.get_single_face_from_boxes(image, single_square_box)
                    singe_face_marks = self.markDetector.detect_marks_on_single_image(single_face_img)
                    mark_detection = self.markDetector.fit_markers_in_single_image(singe_face_marks, 
                                                                                        single_square_box)
                    tmp_trk.predict_marks()
                    tmp_trk.correct_marks(mark_detection)
                    
                    if FaceVar.HEADPOSE_ON:
                        # get single face poses
                        # first get 68 landmarks --> get 6 landmarks --> update with kalman filter
                        # Also get landmarks here to make sure head pose could ran separately
                        '''single_square_box = self.markDetector.square_single_facebox(tmp_trk.box)
                        single_face_img = self.markDetector.get_single_face_from_boxes(image, single_square_box)
                        singe_face_marks = self.markDetector.detect_marks_on_single_image(single_face_img)
                        mark_detection = self.markDetector.fit_markers_in_single_image(singe_face_marks, 
                                                                                            single_square_box)'''
                        marks6_for_pose = self.poseDetector.get_single_face_pose_marks(mark_detection)
                        tmp_pose = self.poseDetector.solve_single_pose(marks6_for_pose)
                        tmp_pose_np = np.array(tmp_pose).flatten()                    
                        tmp_trk.predict_headPose()
                        tmp_trk.correct_headPose(tmp_pose_np)                    
                        tmp_trk.pose_trace.append(tmp_trk.pose_state)
                
                
        # Deal with unmatched detections
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                # Face bonding box
                det = detections[idx]
                self.IdCount +=1
                tmp_trk = Track(det, self.IdCount)
                tmp_trk.predict()                
                self.tracks.append(tmp_trk)
                tmp_trk.trace.append([0.5*(tmp_trk.box[0]+tmp_trk.box[2]),
                                      0.5*(tmp_trk.box[1]+tmp_trk.box[3])])
                # Face landmarks
                if FaceVar.LADNMARK_ON or FaceVar.HEADPOSE_ON:      
                    single_square_box = self.markDetector.square_single_facebox(tmp_trk.box)
                    single_face_img = self.markDetector.get_single_face_from_boxes(image, single_square_box)
                    singe_face_marks = self.markDetector.detect_marks_on_single_image(single_face_img)
                    mark_detection = self.markDetector.fit_markers_in_single_image(singe_face_marks, 
                                                                                        single_square_box)
                    tmp_trk.initialize_landmarks(mark_detection)
                    tmp_trk.marks_trace.append(tmp_trk.marks_state)
                    
                    if FaceVar.HEADPOSE_ON:
                        # get single face poses
                        # first get 68 landmarks --> get 6 landmarks --> update with kalman filter
                        # Also get landmarks here to make sure head pose could ran separately
                        '''single_square_box = self.markDetector.square_single_facebox(tmp_trk.box)
                        single_face_img = self.markDetector.get_single_face_from_boxes(image, single_square_box)
                        singe_face_marks = self.markDetector.detect_marks_on_single_image(single_face_img)
                        mark_detection = self.markDetector.fit_markers_in_single_image(singe_face_marks, 
                                                                                            single_square_box)'''
                        marks6_for_pose = self.poseDetector.get_single_face_pose_marks(mark_detection)
                        tmp_pose = self.poseDetector.solve_single_pose(marks6_for_pose)
                        tmp_pose_np = np.array(tmp_pose).flatten()   
                        tmp_trk.initialize_headPose(tmp_pose_np)
                        tmp_trk.pose_trace.append(tmp_trk.pose_state)
                
        # Deal with unmatched tracks
        if len(unmatched_tracks) >0:
            for trk_idx in unmatched_tracks:
                # Face bonding box
                tmp_trk = self.tracks[trk_idx]
                tmp_trk.num_loss +=1
                tmp_trk.predict()
                tmp_trk.trace.append([0.5*(tmp_trk.box[0]+tmp_trk.box[2]),
                                      0.5*(tmp_trk.box[1]+tmp_trk.box[3])])
                
############################# To do, to change the kalman filter for not detected marks and poses, 
############################# otherwise it predict wrong ones
                # Face landmarks
                if FaceVar.LADNMARK_ON:                   
                    tmp_trk.predict_marks()
                    tmp_trk.marks_trace.append(tmp_trk.marks_state)
                if FaceVar.HEADPOSE_ON:
                    # predict only
                    tmp_trk.predict_headPose()
                    tmp_trk.pose_trace.append(tmp_trk.pose_state)
                    
        # delete no useful lists
        # tracks_to_delete = filter(lambda x: x.num_loss>FaceVar.MAX_AGE, self.tracks)
        self.tracks = [x for x in self.tracks if x.num_loss <= FaceVar.MAX_AGE]
        
            
    def annotate_BBox(self, img):      
        # good_track_list = []
        for trk in self.tracks:
            if trk.hits >= FaceVar.MIN_HITS and trk.num_loss <= FaceVar.MAX_AGE:
                # good_track_list.append(trk)

                if FaceVar.DRAW_DETECTION_BOX:
                    img = draw_box_on_image(img, trk)
                if FaceVar.LADNMARK_ON:
                    img=draw_marks_on_image(img, trk)
                    
                if FaceVar.HEADPOSE_ON:
                    ### draw head pose on images
                    #print(trk.pose_state)
                    tmp_vect = np.array(trk.pose_state).reshape(-1,3)
                    img=self.poseDetector._draw_annotation_box(img, 
                                                               tmp_vect[0], 
                                                               tmp_vect[1], 
                                                               color=(255, 0, 0), 
                                                               line_width=1)
        return img
    


def main():
    detector = FaceDetector()
    files = sorted(glob.glob('./data/face_sequence/*.jpg'))
    images_seq = [cv2.imread(file) for file in files]
    
    tracker = Tracker(FaceVar.IOU_THRESHOLD, img_size=(720, 1280))
    index=1
    for image in images_seq:
        start = time.time()
        conf, boxes = detector.get_faceboxes(image=image, threshold=0.9)
       
        #print(boxes)
        matches, unmatched_dets, unmatched_tracks = tracker.assign_detections_to_trackers(boxes)
        #print('matches: ',matches)
        #print('unmatched_dets: ', unmatched_dets)
        #qq('unmatched_tracks: ', unmatched_tracks)
        tracker.update(image, boxes, matches, unmatched_dets, unmatched_tracks)
        
        image = tracker.annotate_BBox(image)
        if FaceVar.DRAW_ORIG_BBOX:
            image = draw_BBox(image=image, faceboxes=boxes, confidences=conf) 
        cv2.putText(image, 'Frame: '+str(index), (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
        #cv2.imwrite('image_seq'+str(index)+'.jpg', image)
        print(time.time()-start)
        cv2.imshow('image_seq'+str(index), image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        index +=1
    
if __name__ == '__main__':
    main()
    
                
    
                
            
    
        
        