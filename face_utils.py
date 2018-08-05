# utils
import numpy as np
import cv2

def box_iou2 (a, b):
  '''
  Calculate the ratio between intersection and the union of
  two boxes a and b
  a[0], a[1], a[2], a[3] <-> left, up, right, bottom
  Return: I, A, B
    where
      I is the intersection
      A is the other slide of boxA in the Venn diagram
      B is the other slide of boxB in the Venn diagram
    The Track-Det association condition must include:
      IOU = I / (A+I+B) must > thresh e.g. 0.3
      error_A = A / (A+I) must < thresh e.g. 0.5
      error_B = B / (B+I) must < thresh e.g. 0.5
  '''
  smooth=1.
  w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
  h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
  I = w_intsec * h_intsec
  areaA = (a[2] - a[0])*(a[3] - a[1])
  areaB = (b[2] - b[0])*(b[3] - b[1])
#  A = areaA - I
#  B = areaB - I

  return float(I)/(areaA + areaB -I + smooth)
  #return I, A, B

#============draw face tracking annotations on images==========

def draw_box_on_image(image, track):
        ''''Draw the tracking results on a single image. 
        Input: image -- a image
                track -- a single track object'''
        box_t = track.box
        Id = track.trackId
 
        cv2.rectangle(image, (box_t[0], box_t[1]), (box_t[2], box_t[3]), (0,0,255))                  
        cv2.putText(image, str(Id), (box_t[0], box_t[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        return image
    
def draw_marks_on_image(image, track, color=(0,255,0)):
    '''draw face landmarks results on a single image
    Input: image -- a image track -- a single track object'''
    for mark in track.marks_state:  
        # print(mark)              
        cv2.circle(image, (int(mark[0]), int(mark[1])), 
                       1, color, -1, cv2.LINE_AA)            
    return image

#def draw_Bbox(image, Bbox):
#        ''''Draw the tracking results on a single image. 
#        Input: image -- a image
#                track -- a single track object'''        
# 
#    cv2.rectangle(image, (Bbox[0], Bbox[1]), (Bbox[2], Bbox[3]), (0,0,255))
#    return image

def draw_BBox(image, faceboxes, confidences):
    '''Draw the originally detected bounding boxes on image from the face detector.
    Inpit: image --- the image where the detection come from
        faceboxes --- face bonding box from the face detector
        confidences --- confidence score of the face detection'''
    
    for facebox, confidence in zip(faceboxes, confidences):
        cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (210,250,250))
        label= "face: %.4f" % confidence
            #label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
#            cv2.rectangle(image, (facebox[0], facebox[1]-label_size[1]), 
#                          (facebox[0]+label_size[0],
#                           facebox[1]+base_line),
#                           (0,255,0), cv2.FILLED)
            #cv2.circle(image, (facebox[0], facebox[1]), 3, (255,255,255))
            #cv2.circle(image, (facebox[0], facebox[1]+100), 3, (255,255,255))
        cv2.putText(image, label, (facebox[2]-110, facebox[3]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210,250,250))
    return image