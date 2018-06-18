'''Face detection using Resnet-SSD detection network
object detection, resnet, single shot multi-box detection
https://arxiv.org/pdf/1702.01243.pdf
modified from work by yinguobing
'''
import time
import cv2
import numpy as np
import tensorflow as tf

class FaceDetector:
    '''Detect faces in an image'''
    def __init__(self,
                 dnn_proto_text='models/face_detector/res10_300x300_ssd_iter_140000.prototxt',
                 dnn_model='models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None
  
    
    def get_faceboxes(self, image, threshold=0.5):
        '''Get the bounding box of faces in image using dnn'''
        rows, cols, _ = image.shape
        confidences = []
        faceboxes= []
        
        self.face_net.setInput(cv2.dnn.blobFromImage(image=image,
                                             scalefactor=1.0,
                                             size=(300,300),
                                             mean=(104.0, 177.0, 123.0),
                                             swapRB=False,
                                             crop=False))
        detections = self.face_net.forward()
        
        for result in detections[0,0,:,:]:
            confidence=result[2]
            if confidence > threshold:
                x_left_top = int(result[3]*cols)
                y_left_top = int(result[4]*rows)
                x_right_bottom = int(result[5]*cols)
                y_right_bottom = int(result[6]*rows)
                confidences.append(confidence)
                faceboxes.append([x_left_top, y_left_top, 
                                  x_right_bottom, y_right_bottom])
        self.detection_result=[faceboxes, confidences]
        #print(self.detection_result)
        return confidences, faceboxes
    
    def draw_all_results(self, image):
        '''Draw the detected bounding boxes on image.'''
        faceboxes, confidences = self.detection_result
        for facebox, confidence in zip(faceboxes, confidences):
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0,255,0))
            label= "face: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,1)
            cv2.rectangle(image, (facebox[0], facebox[1]-label_size[1]), 
                          (facebox[0]+label_size[0],
                           facebox[1]+base_line),
                           (0,255,0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

def main():
    start_time = time.time()
    detector = FaceDetector()
    process_start = time.time()
    print('time to initiate FaceDetector: ', process_start-start_time)
    filepath = '/home/eric/Documents/face_analysis/data/photos/xin.jpg'
    img = cv2.imread(filepath)
    #img = cv2.resize(img, (300,300))    
    conf, box = detector.get_faceboxes(image=img, threshold=0.9)
    detector.draw_all_results(img)
    print('time to finish processing', time.time()-process_start)
    cv2.imshow('image',img)
    print(time.time()-process_start)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
        
if __name__ == '__main__':
    main()
        
        