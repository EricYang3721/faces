'''Estimate head pose by the facial landmarks
modified from work by yinguobing'''
import numpy as np
import cv2
import time
class PoseEstimator:
    '''Estimate head pose by the facial landmarks'''
    
    def __init__(self, img_size=(480, 640)):
        self.size = img_size
        
        # 3D model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5
    
        self.model_points_68 = self._get_full_model_points()
        
        # Cameral parameters
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1]/2, self.size[0]/2)
        self.camera_matrix = np.array(
                [[self.focal_length, 0, self.camera_center[0]],
                 [0, self.focal_length, self.camera_center[1]],
                 [0,0,1]], dtype="double")
        
        # Assuming no lens distortion
        self.dist_coeffs = np.zeros((4,1))
        
        # Rotation vector and translation vector
        # Initialize rotation and translation vectors for iterative solving cv2.PnP
        #self.r_vec = None
        #self.t_vec = None
#        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
#        self.t_vec = np.array(
#            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

        
    def _get_full_model_points(self, filename='models/model_landmark.txt'):
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:, -1] *=-1            
        return model_points
    
    def solve_pose(self, image_points):
        '''get the roation and translation vectors'''
        rotation_vectors=[]
        translation_vectors=[]
        for face_point in image_points:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                    self.model_points, 
                    face_point, 
                    self.camera_matrix, self.dist_coeffs)
            rotation_vectors.append(rotation_vector)
            translation_vectors.append(translation_vector)
        return (rotation_vectors, translation_vectors)
    
    def solve_pose_by_68_points(self, image_points):
        '''Solve pose from all 68 image points return (rotation_vector, 
        translation_vector) as pose'''
        rotation_vectors=[]
        translation_vectors=[]        
        for face_point in image_points:
        # initialize the translation and rotation vector for solving the final ones
            #if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_68,
            face_point, self.camera_matrix, self.dist_coeffs)
#            self.r_vec = rotation_vector
#            self.t_vec = translation_vector
            
            
#            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_68,
#            face_point, self.camera_matrix, self.dist_coeffs, rvec=self.r_vec, tvec=self.t_vec,
#            useExtrinsicGuess=True)
            
            rotation_vectors.append(rotation_vector)
            translation_vectors.append(translation_vector)
        return (rotation_vectors, translation_vectors)
    
    def get_pose_marks(self, marks):
        '''get 6 points out of 68 detected ones to match the ones in self.model_points'''
        pose_marks=[]
        for mark in marks:
            temp_pose_marks=[]
            temp_pose_marks.append(mark[30])    # Nose tip
            temp_pose_marks.append(mark[8])     # Chin
            temp_pose_marks.append(mark[36])    # Left eye left corner
            temp_pose_marks.append(mark[45])    # Right eye right corner
            temp_pose_marks.append(mark[48])    # Left Mouth corner
            temp_pose_marks.append(mark[54])    # Right mouth corner
            pose_marks.append(temp_pose_marks)
        return np.array(pose_marks)
    
    def draw_boxes(self, image, rotation_vctors, translation_vectors, color=(0,255,0), line_width=1):
        for r_vec, t_vec in zip(rotation_vctors, translation_vectors):
            self._draw_annotation_box(image, r_vec, t_vec, color=color, line_width=line_width)
    
    
    def _draw_annotation_box(self, image, rotation_vector, translation_vector, color=(0, 255, 0), line_width=1):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        #print(type(self.dist_coeffs))
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

def main():
    
    from MarkDetector import MarkDetector
    markDetector = MarkDetector()
    pose = PoseEstimator()
    start_time = time.time()
    filepath = '/home/eric/Documents/face_analysis/data/photos/group.jpg'
    img = cv2.imread(filepath)
    
    
    boxes = markDetector.extract_cnn_facebox(image=img)
    face_imgs = markDetector.get_face_for_boxes(img, boxes)
    marks = markDetector.detect_marks(face_imgs)
    marks = markDetector.fit_markers_in_image(marks, boxes)    
    
    
    
    ### 2 different function: choose first or last 2    
    #r_vect, t_vect = pose.solve_pose_by_68_points(marks)
    marks = pose.get_pose_marks(marks)    
    pp = pose.solve_pose(marks)
    r_vect, t_vect = pp
    pose_np = np.array(pp).flatten()
    print(pp)
    print(pose_np)
    stabile_pose = np.reshape(pose_np, (-1, 3))
    print(stabile_pose)
    pose.draw_boxes(img, r_vect, t_vect)
    #MarkDetector.draw_marks(image=img, marks=marks)
    #detector.draw_all_results(img)
    
    cv2.imshow('image',img)
    #print(time.time()-start_time)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()
        