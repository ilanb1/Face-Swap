from utils import ImageFace
import cv2
import numpy as np
import time

#preparing the faces:
images_list = []
for index in range(1,6):
    image_path = f"face{index}.jpg"
    image = cv2.imread(image_path)
    images_list.append(image)

#font details for printing the fps:
font = cv2.FONT_HERSHEY_SIMPLEX # font
org = (50, 50) # org
fontScale = 1 # fontScale
color = (255, 0, 255) #  color in BGR
thickness = 2 # Line thickness of 2 px


#Creating the objects
source_image = ImageFace(is_source = True)
destination_image = ImageFace(is_source = False)

#Adding the source image, (can be changed later inside the video loop)
source_image.update_image(images_list[0])


cap = cv2.VideoCapture(0)

# used to record the time at which we processed current frame
prev_frame_time = 0
new_frame_time = 0

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    destination_image.update_image(frame)
    face_swapped = source_image.apply_face(destination_image)

    # Calculating and showing the fps:
    new_frame_time = time.time()
    fps = int( 1 / (new_frame_time - prev_frame_time) )
    prev_frame_time = new_frame_time
    frame = cv2.putText(face_swapped, f'fps: {fps}', org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('frame', face_swapped)


    key = cv2.waitKey(1)

    # Press 'q' to exit the program.
    if key == ord("q"):
        break

    # press 's' to switch to other random source face.
    elif key == ord("s"):
        new_index = np.random.randint(len(images_list))
        new_image = images_list[new_index]
        source_image.update_image(new_image)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()