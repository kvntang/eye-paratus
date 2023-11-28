import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np



# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
video_file_fps = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0
while cap.isOpened():
    success, opencv_frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Resize the frame to 320x320
    resized_frame = cv2.resize(opencv_frame, (320, 320))


    #opencv_frame = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)


    # Display the original image
    cv2.imshow('MediaPipe Object Detection', resized_frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

    frame_index += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
