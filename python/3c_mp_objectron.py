import cv2
import mediapipe as mp

# Initialize MediaPipe Object Detection.
mp_objects = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
object_detection = mp_objects.ObjectDetection(
    model_selection=0, min_detection_confidence=0.5)

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

# Process each frame.
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = object_detection.process(image)

    # Draw the detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

    # Display the resulting frame.
    cv2.imshow('MediaPipe Object Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit.
        break

# Release the webcam and close OpenCV window.
cap.release()
cv2.destroyAllWindows()
