from utils import ImageFace
import cv2

#Lets say we want the face from image1 pasted on the head of image2
image1 = cv2.imread("face5.jpg")
image2 = cv2.imread("face1.jpg")

#Creating the objects
source_image = ImageFace(is_source = True)
destination_image = ImageFace(is_source = False)

#Adding the images:
source_image.update_image(image1)
destination_image.update_image(image2)

#Getting the new image and plotting it
face_swapped = source_image.apply_face(destination_image)
cv2.imshow("image2 with the face of image1" , face_swapped)
cv2.waitKey(0)

