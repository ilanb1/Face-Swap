import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt

# OOP implementation:

class ImageFace:


    def __init__(self, is_source=True):
        # During the initiation, we need to specify if the object is source or destination,
        # i.e whether the face will be taken from it or pasted on it respectively.
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.is_source = is_source

    def update_image(self, image):
        self.image = image

        #adding (or updating) the image. If the object is the destination, only the required calculations will be made.
        if self.find_face():
            self.get_landmarks()
            self.get_face_boundingRect()

            if self.is_source:
                self.get_delaunay_triangulation()
                self.get_triangles_boundingRects()

        elif self.is_source:
            raise Exception("Couldn't find any face in that source image, please try other image.")


    def find_face(self):

        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(self.gray_image)
        if len(faces) == 0 :
            self.face = None
            return False
        self.face = faces[0]
        return True

    def get_landmarks(self):
        #finding the 68 landmark points on the face.
        landmarks = self.predictor(self.gray_image, self.face)
        self.landmark_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]


    def get_face_boundingRect(self):
        self.face_mask = np.zeros_like(self.gray_image)
        convexhull = cv2.convexHull(np.array(self.landmark_points, np.int32))
        cv2.fillConvexPoly(self.face_mask, convexhull, 255)
        self.face_boundingRect_position = cv2.boundingRect(convexhull)
        (x, y, w, h) = self.face_boundingRect_position
        self.face_boundingRect = self.image[y: y + h, x: x + w]
        self.cropped_landmarks = [(a - x, b - y) for (a, b) in self.landmark_points]

    def get_delaunay_triangulation(self):
        rect = (0, 0, self.face_boundingRect.shape[1], self.face_boundingRect.shape[0])
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(self.cropped_landmarks)
        triangles = subdiv.getTriangleList()
        triangles = [[[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]] for t in triangles]
        self.triangles = np.array(triangles, dtype=np.int32)

        # vertices_indexes:
        points_indexes = [np.where((self.cropped_landmarks == vertex).all(axis=1)) for triangle in self.triangles for
                          vertex in triangle]
        self.vertices_indexes = list(np.reshape(points_indexes, (-1, 3)))

    def get_triangles_boundingRects(self):
        cropped_triangles = []
        updated_vertices = []
        for t in self.triangles:
            (x, y, w, h) = cv2.boundingRect(t)
            cropped_rect = self.face_boundingRect[y:y + h, x: x + w]
            triangle_updated_vertices = np.array([[a - x, b - y] for [a, b] in t])

            cropped_triangles.append(cropped_rect)
            updated_vertices.append(triangle_updated_vertices)
        self.cropped_triangles = np.array(cropped_triangles, dtype=object)
        self.updated_vertices = np.array(updated_vertices)


    # Transfers the face from the source object to the destination object.
    def apply_face(self, image_object):
        if image_object.face is None:
            return image_object.image
        img = image_object.face_boundingRect.copy()
        for triangle_index, triangle_vertices_indexes in enumerate(self.vertices_indexes):
            points = [image_object.cropped_landmarks[i] for i in triangle_vertices_indexes]
            (x, y, w, h) = cv2.boundingRect(np.array([points]))
            triangle_updated_vertices = np.array([[a - x, b - y] for [a, b] in points])
            triangle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(triangle_mask, np.array([triangle_updated_vertices]), 255)
            M = cv2.getAffineTransform(np.array(self.updated_vertices[triangle_index], np.float32),
                                       np.array(triangle_updated_vertices, np.float32))
            warped_triangle = cv2.warpAffine(self.cropped_triangles[triangle_index], M, (w, h))
            cv2.bitwise_and(warped_triangle,warped_triangle, dst = img[y:y + h, x: x + w] , mask =triangle_mask )

        (x, y, w, h) = image_object.face_boundingRect_position
        final = image_object.image.copy()
        cv2.bitwise_and(img, img, dst = final[y:y + h, x: x + w], mask = image_object.face_mask[y:y + h, x: x + w])
        return final



    #########################################################
    # The below methods are not necessary for the face swap,
    # they are just for visualizing any requested stage of the process for research purposes.
    ###########################################################

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def plot_landmarks(self):
        img = self.image.copy()
        for i in range(68):
            cv2.circle(img, center=tuple(self.landmark_points[i]), radius=3, thickness=-1, color=(255, 0, 0))
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.show()

    def plot_cropped_face(self):
        plt.imshow(self.face_boundingRect)
        plt.show()

    def plot_triangles(self):
        img = self.face_boundingRect.copy()
        for t in self.triangles:
            cv2.polylines(img, pts=[t], isClosed=True, color=(0, 0, 255), thickness=1)
        # plt.figure(figsize = (12,12))
        plt.imshow(img)
        plt.show()

