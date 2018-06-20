from multiprocessing import Process, Queue
import numpy as np
from MarkDetector import MarkDetector
from PoseEstimator import PoseEstimator
from FaceDetector import FaceDetector
import cv2

CNN_INPUT_SIZE = 128 # constant for mark detector input size

def get_face(detector, img_queue, box_queue):
    '''Get face from image queue. Multiprocessing'''
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)
    
def main():
    # get video from webcam or video file
    video_src = 0  # 0 means webcam; set file name if video
    cam = cv2.VideoCapture(video_src)
    _, sample_frame = cam.read()  # get 1 sample frame to setup codes
    
    # Initialize MarkDetector
    mark_detector = MarkDetector()
    
    # Setup process and queues for multiprocessing
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(mark_detector,
                                                 img_queue, 
                                                 box_queue))
    box_process.start()
    
    # Initialize PoseEstimator to solve pose. Use 1 sampe frame to setup
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))
    
    while True:
        frame_got, frame = cam.read()
        if frame_got is False:
            break
        
        # crop image if needed
        # frame = cv2.resize(frame, [500, 300])
        
        # flip image if needed in webcam
#        if video_src ==0:
#            frame = cv2.flip(frame, 2)
        
        # 3 Steps to estimate pose
        # 1. detect face
        # 2. detect landmarks
        # 3. estimate pose
        
        # Feed frame to image queue
        img_queue.put(frame)
        
        # Get facebox from box queue
        facebox = box_queue.get()
        
        if facebox is not None:
            # get face images from the face boxes and images
            faces = mark_detector.get_face_for_boxes(frame, facebox)
            
            # get markers from face images
            marks = mark_detector.detect_marks(faces)
            marks = mark_detector.fit_markers_in_image(marks, facebox)    
        
            # Draw markers if necessary
            MarkDetector.draw_marks(image=frame, marksFace=marks)
            
            # Solve pose by 68 points
#            r_vect, t_vect = pose_estimator.solve_pose_by_68_points(marks)
            
            # Solve pose by 6 points             
            marks = pose_estimator.get_pose_marks(marks)
            r_vect, t_vect = pose_estimator.solve_pose(marks)

                
            
            # Draw pose boxes on the image
            pose_estimator.draw_boxes(frame, r_vect, t_vect)
        
        cv2.imshow("preview", frame)
        if cv2.waitKey(10) == 27:
            break
    
    # clean up the multiprocessing processes
    box_process.terminate()
    box_process.join()
        
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
