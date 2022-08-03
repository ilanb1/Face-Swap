**Face Swapping Project**

Face swapping between 2 images (can also be used for real-time video).

This is my OOP implementation from scratch to the tutorial at: <br>
https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/

The steps are:

•Finding the face in the image.
•finding 68 landmark points on the face.
•on the source image: Doing Delaunay triangulation according to the landmark points.
•on the destination image: Doing triangulation according to the same vertices indexes of the source image triangles.
•performing "Affine transformation" to each of the source's triangles so it will match the destination's triangles shape.
•pasting each of the new triangles to their correct position in the destination image.
•The last step is usually doing "Seamless Cloning" to match the color of the face and the head, but our application is designed also for animal faces, so it's important to keep the original face color, so this step is not implemented. 







