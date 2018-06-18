'''Landmark detector by Convolutional neural network
modified from work by yinguobing'''
import cv2
import tensorflow as tf
import numpy as np
from FaceDetector import FaceDetector
import time

class MarkDetector:
    def __init__(self, mark_model='models/landmark_detector/frozen_inference_graph.pb'):
        self.face_detector = FaceDetector()
        
        self.cnn_input_size=128
        self.marks = None
        
        # Get a tensorflow session ready for landmark detection
        # load a frozen tensorflow model into memory
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(mark_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph=detection_graph
        self.sess = tf.Session(graph=detection_graph)
        
    @staticmethod
    def draw_box(image, boxes, box_color=(255,255,255)):
        '''Draw square boxes on images'''
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]),
                          (box[2], box[3]), box_color)
            
    @staticmethod
    def move_box(box, offset):
        '''Move the boxt to direction specified by a vector offset'''
        leftT_x = box[0] + offset[0]
        leftT_y = box[1]+offset[1]
        rightB_x = box[2]+offset[0]
        rightB_y = box[3] + offset[1]
        return [leftT_x, leftT_y, rightB_x, rightB_y]        
    
    @staticmethod
    def get_square_box(box):
        '''Get a square box out by expanding a given box'''
        leftT_x = box[0]
        leftT_y = box[1]
        rightB_x = box[2]
        rightB_y = box[3]
        
        # Get the width and height
        box_width = rightB_x - leftT_x
        box_height = rightB_y - leftT_y
        
        # check if it is already a square. if not, make it into a square
        diff = box_height-box_width
        delta = int(abs(diff)/2)
        
        if diff == 0:
            return box
        elif diff > 0: # width < height, expand width
            leftT_x -= delta
            rightB_x += delta
            if diff%2==1:
                rightB_x +=1
        else: # width > height, expand height
            leftT_y -= delta
            rightB_y += delta
            if diff%2==1:
                rightB_y +=1
        
        # make sure the box is always squre
        assert ((rightB_x-leftT_x) == (rightB_y - leftT_y)), 'Box is not square.'
        return [leftT_x, leftT_y, rightB_x, rightB_y]
    
    @staticmethod
    def box_in_image(box, image):
        '''check if the box could fit in the image'''
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >=0 and box[1] >=0 and box[2] <=cols and box[3] <=rows
    
    
    def extract_cnn_facebox(self, image):
        """Extract face area from image. Returns the box coordinates"""
        _, raw_boxes = self.face_detector.get_faceboxes(image, threshold=0.9)
        boxes=[]
        for box in raw_boxes:
            diff_height_width = (box[3]-box[1]) - (box[2]-box[0])
            offset_y = int(abs(diff_height_width/2))
            box_moved = self.move_box(box, [0, offset_y])
            
            # Make box square
            facebox = self.get_square_box(box_moved)
            
            # if box is out of image
            ############# need to modify if detect multiple faces
           # if self.box_in_image(facebox, image):
            boxes.append(facebox)
            
        return boxes
    
    def detect_marks(self, images_np):
        '''Detect makrs from cropped face image. 
        Returns the coordiate/width or length of cropped image __ image_np'''
        #Get result tensor by its name
        logits_tensor = self.graph.get_tensor_by_name('logits/BiasAdd:0')
        all_marks = []
        for image in images_np:
            # Actual detection
            predictions = self.sess.run(
                logits_tensor,
                feed_dict={'input_image_tensor:0':image})
        
            # Convert preditions to landmarks
            marks = np.array(predictions).flatten()
            marks = np.reshape(marks, (-1, 2))
            all_marks.append(marks)
        #print(marks)
        return all_marks
    
    @staticmethod
    def draw_marks(image, marksFace, color=(0,255,0)):
        '''Draw mark points on the image'''
        rows = image.shape[0]
        cols = image.shape[1]
        for marks in marksFace:
            for mark in marks:
                if(mark[0]<0 or mark[0]>=cols or mark[1]<0 or mark[1]>=rows):
                    continue
                cv2.circle(image, (int(mark[0]), int(mark[1])), 
                       1, color, -1, cv2.LINE_AA)
    
    def get_face_for_boxes(self, image, boxes):
        '''crop the image with given box, make it ready for marker detection'''
        
        ######### Problem to fix if face is out of box
        face_images = []
        for box in boxes:
            if not self.box_in_image(box, image):                
                face_image = self.crop_at_image_boarder(image, box)
            else: 
                face_image = image[box[1]:box[3], box[0]:box[2]]
            face_image = cv2.resize(face_image, (128,128))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_images.append(face_image)
        return face_images
    
    @staticmethod
    def fit_markers_in_image(markers, boxes):
        for marker, box in zip(markers, boxes):
            marker *=(box[2]-box[0])
            marker[:, 0] += box[0]
            marker[:,1] += box[1]
        return markers    
    
    @staticmethod
    def crop_at_image_boarder(image, box):
        '''when cropping image with box, part of croped image could be out of
        image edge, so pad it'''
        rows = image.shape[0]
        cols = image.shape[1]
        delta_left =0
        delta_right =0
        delta_top = 0
        delta_bottom = 0

        if box[0] < 0:
            delta_left = -box[0]
        if box[2] >= cols:
            delta_right = box[2] - cols + 1
        if box[1] < 0:
            delta_top = -box[1]
        if box[3] >= rows:
            delta_bottom = box[3] - rows +1        
        cropped = image[max(0, box[1]):min(rows-1, box[3]) ,
                        max(0, box[0]):min(cols-1, box[2])]

        pad_w = ((delta_top, delta_bottom), (delta_left, delta_right), (0, 0))
        cropped = np.lib.pad(cropped, pad_w, 'edge')       
        return cropped
        
        
        
# Question remains --> need to modify extract_cnn_faces if for multiple faces   
def main():
    start_time = time.time()
    markDetector = MarkDetector()
    process_start = time.time()
    print('time to initiate FaceDetector: ', process_start-start_time)
    filepath = '/home/eric/Documents/face_analysis/data/photos/group.jpg'
    img = cv2.imread(filepath)    
    
    boxes = markDetector.extract_cnn_facebox(image=img)
    face_imgs = markDetector.get_face_for_boxes(img, boxes)
    
    
    marks = markDetector.detect_marks(face_imgs)
    marks = markDetector.fit_markers_in_image(marks, boxes)
    MarkDetector.draw_marks(image=img, marksFace=marks)
    print('time to finish processing', time.time()-process_start)
    #detector.draw_all_results(img)
    cv2.imshow('image',img)
    print(time.time()-process_start)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()
        