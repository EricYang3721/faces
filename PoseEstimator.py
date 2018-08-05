'''Estimate head pose by the facial landmarks
modified from work by yinguobing'''
import numpy as np
import cv2
import time
class PoseEstimator:
    '''Estimate head pose by the facial landmarks by mapping the detected facial landmarks
    to the landmar locations of an average face.
    2 methods are available here. It could use 6 landmarks (self.model_points) or all
    68 landmarks (self.model_points_68)'''
    
    def __init__(self, img_size=(480, 640)):
        #initialzize image size
        self.size = img_size
        
        # 3D model points for the average face (6 points);
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5
        
        # 3D model points for average face(68points)
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
        ''' To obtain the coordinates of 68 landmarks for the average face.
        Input: filename --- the path to the txt file
        Output: model_points --- 68 by 3 Numpy array of the landmarks
        '''
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:, -1] *=-1            
        return model_points
    
    def solve_pose(self, image_points):
        '''get the roation and translation vectors from the detected 2D landmarks and 
        3D landmarks of the average face, and camera parameters. For multiple faces at
        the same time.
        Input: image_points --- 2D landmarks from landmark detector. List of n faces, each face
            with 68 points, each points with 2 entries [x, y].
        Output: rotation_vectors --- list of rotation vectors for mapping 3D points to 2D plane for all faces.
        Translation_vectors --- list of translation vectors for mapping 3D points to 2D plane for all faces.'''
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
    
    def solve_single_pose(self, face_point):
        '''get the roation and translation vectors for a single face.
        Input: face_point --- the landmarks (6 or 68) used to find the rotation and translation 
                            vectors
        Output: rotation_vector --- rotation vector for mapping 3D points on 2D plane, single face
                translation_vector --- translation vector for mapping 3D points on 2D plane, single face'''
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
                    self.model_points, 
                    face_point, 
                    self.camera_matrix, self.dist_coeffs)

        return (rotation_vector, translation_vector)
    
    def solve_pose_by_68_points(self, image_points):
        '''Solve multiple poses from all 68 image points return (rotation_vector, 
        translation_vector) as pose
        Input: image_points --- list of n faces, each face contains 68 2D landmars points
        Output: rotation_vectors --- list of rotation vectors mapping 3D points on 2D plane, multiple faces
                translation_vectors --- list of translation vectors mapping 3D points on 2D plane, multiple faces'''
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
        '''get 6 points out of 68 detected ones matching the ones in self.model_points (6 3D points), 
        the results are used to mapping from 3D to 2D plane.
        Input: marks --- list of landmarks containing n faces. Each face has 68 landmakrs
        Output: np.array(pose_marks) --- list of landmarks containing n faces, each face has 6 landmarks 
        '''
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
    
    def get_single_face_pose_marks(self, face_mark):
        '''get 6 points out of 68 detected ones to match the ones in self.model_points, 
        only for a single face
        Input: face_mark --- a list of 68 landmarks for a single face
        Output: np.array(pose_marks) --- a list of 6 landmarks for a single face'''
        
        pose_marks=[]             
        pose_marks.append(face_mark[30])    # Nose tip
        pose_marks.append(face_mark[8])     # Chin
        pose_marks.append(face_mark[36])    # Left eye left corner
        pose_marks.append(face_mark[45])    # Right eye right corner
        pose_marks.append(face_mark[48])    # Left Mouth corner
        pose_marks.append(face_mark[54])    # Right mouth corner        
        return np.array(pose_marks)
    
    def draw_boxes(self, image, rotation_vctors, translation_vectors, color=(0,255,0), line_width=1):
        '''draw annotation boxes for multiple faces on an image. The each annotation box indicate pose of
        each head. Simply by calling the function to drawn each face with its mapping (transaltion and rotation vector).
        Input: image --- the image to be drawn on
            rotation_vectors --- list of mapping from 3D to 2D plane containing n vectors for n faces
            translation_vectors --- list of mapping from 3D to 2D plane containing n vectors for n faces
            '''
        for r_vec, t_vec in zip(rotation_vctors, translation_vectors):
            # get translation and rotation vector to drawn head pose for individual faces 
            self._draw_annotation_box(image, r_vec, t_vec, color=color, line_width=line_width)
    
    
    def _draw_annotation_box(self, image, rotation_vector, translation_vector, color=(0, 255, 0), line_width=1):
        """Draw a 3D box as annotation of head pose for a single face. Each head pose annotation contains
        a smaller square on face surface, and a larger sqaure in front of the smaller one. The openning
        of the 2 squares indicate the orientation of head pose.
        Input: 
            image --- image to be drawn on.
            rotation_vector --- a mapping vector for a single face from 3D to 2D plane (rotation)
            translation_vector --- a mapping vector for single face from 3D to 2D plane (translation)
        Output: image --- the annotated image
            """
        point_3d = []  # a list to save the 3D points to show head pose
        rear_size = 50 # smaller square edge length
        rear_depth = 0 # distance between small squares to nose tip
        point_3d.append((-rear_size, -rear_size, rear_depth)) # get all 4 points for small squared
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100 # larger square edge length
        front_depth = 50 # distance between large squares to nose tip
        point_3d.append((-front_size, -front_size, front_depth)) # all 4 points for larger square
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
        point_2d = np.int32(point_2d.reshape(-1, 2)) # convert to integer for pixels
        # print('2D points', point_2d)
        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA) 
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
        return image
    
    def _draw_annotation_arrow(self, image, rotation_vector, translation_vector, color=(0, 255, 0), line_width=1):
        '''draw arrow as the head pose direction (direction facing to), single face only.
        Input: 
            image --- image to be drawn on.
            rotation_vector --- a mapping vector for a single face from 3D to 2D plane (rotation)
            translation_vector --- a mapping vector for single face from 3D to 2D plane (translation)
        Output: image --- the annotated image
        '''
        points_3D=[] # a list to store the 3D points to draw
        rear_point_3D = [0,0,0] # the rear point for the arrow
        front_point_3D = [0,0,100] # the point for the tip of the array
        points_3D.append(rear_point_3D)
        points_3D.append(front_point_3D)
        points_3D = np.array(points_3D, dtype=np.float).reshape(-1, 3)
        # map the 3D points onto 2D image plane
        (points_2d, _) = cv2.projectPoints(points_3D,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeffs)
        points_2d = np.int32(points_2d.reshape(-1, 2)) # convert to integer
        # draw on image plane
        cv2.arrowedLine(image, tuple(points_2d[0]), tuple(points_2d[1]), (255,0,0), 2, tipLength=0.5)
        return image
        
def main():
    
    from MarkDetector import MarkDetector
    markDetector = MarkDetector()
    pose = PoseEstimator()
    start_time = time.time()
    filepath = '/home/eric/Documents/face_analysis/data/photos/group.jpg'
    img = cv2.imread(filepath)    
    
    boxes = markDetector.extract_cnn_facebox(image=img)  # extract bonding boxes
    face_imgs = markDetector.get_face_for_boxes(img, boxes) # extract face images by bonding boxes
    marks = markDetector.detect_marks(face_imgs)  # detect landmarks on the cropped images
    marks = markDetector.fit_markers_in_image(marks, boxes) # get the landmark coordinates on the original images
    
    
    
    ### 2 different function: choose first or last 2    
    #r_vect, t_vect = pose.solve_pose_by_68_points(marks)
    marks = pose.get_pose_marks(marks)    # select 6 points out of 68
    pp = pose.solve_pose(marks)   # get the mapping from 3D points to 2D points
#    tt = pose.solve_single_pose(marks[0]) # debug single face function
#    print(tt)
#    print(np.array(tt).flatten())
    r_vect, t_vect = pp
    pose_np = np.array(pp).flatten() 
    #print(pp)
    #print(pose_np)
#    stabile_pose = np.reshape(pose_np, (-1, 3))
    #print(stabile_pose)
    pose.draw_boxes(img, r_vect, t_vect) # draw annotation for head pose on image
    #MarkDetector.draw_marks(image=img, marks=marks)
    #detector.draw_all_results(img)
    
    cv2.imshow('image',img)
    #print(time.time()-start_time)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()
        