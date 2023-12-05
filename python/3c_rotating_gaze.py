#LOGIC
    #camera on
    #generate random new coordinate
    #move to new coordinate
        #during movement perform object detection + projection
        #if something is interesting/detected (choose one of the detection)
            #log current angle
            #set new coordinate (maybe have a deceleration(midway) coord, then to new coord )
            #
            #ACTIVE SEEING: focus on the object (singulate the projection)
            #project "thinking"
        #else: 
            #set new coord immediately
    #move on
        #PASSIVE SEEING: project on ALL detected objects


import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    if detection.categories[0].score < 0.2:
       continue

    # Draw bounding_box
    bbox = detection.bounding_box
    x_scale = 400/320
    y_scale = 224/320
    scaled_bbox_origin_x = round(bbox.origin_x * x_scale)
    scaled_bbox_origin_y = round(bbox.origin_y * y_scale)
    scaled_bbox_width = round(bbox.width * x_scale)
    scaled_bbox_height = round(bbox.height * y_scale)
    start_point = scaled_bbox_origin_x, scaled_bbox_origin_y
    end_point = scaled_bbox_origin_x + scaled_bbox_width, scaled_bbox_origin_y + scaled_bbox_height
    cv2.rectangle(image, start_point, end_point, RECT_COLOR, -1)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + scaled_bbox_origin_x,
                     MARGIN + ROW_SIZE + scaled_bbox_origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

##########################################################################################################
##########################################################################################################

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
RECT_COLOR = (255, 0, 0) # red
TEXT_COLOR = (255, 255, 255)  # white

prev_time = 0

# Create a window
cv2.namedWindow("fullscreen", cv2.WND_PROP_FULLSCREEN)

# Set window to fullscreen
cv2.setWindowProperty("fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#Object Detector Setup
model_path = 'C:/Users/Kevin/Documents/GitHub/machine-gaze/efficientdet_lite0.tflite'
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.VIDEO)
detector = ObjectDetector.create_from_options(options)

#Initialize OpenCV
cap = cv2.VideoCapture(0)
video_file_fps = cap.get(cv2.CAP_PROP_FPS)

#Main Video Loop
frame_index = 0
while cap.isOpened():
    curr_time = time.time()
    if curr_time - prev_time < 0.1:
       continue
    else:
       prev_time = curr_time
    
    success, opencv_frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #openCV Setup
    opencv_frame = opencv_frame[140:364,123:523] #custom resizing, match with projector specs
    cv2.imshow("fullscreen", opencv_frame)
    rgb_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB) #BGR to RGB
    resized_frame = cv2.resize(opencv_frame, (320, 320))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_frame)
    frame_timestamp_ms = int(1000 * frame_index / video_file_fps)

    #detection + visualize + project
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    image_copy = np.full(opencv_frame.shape, 255, dtype=np.uint8)   # white
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("fullscreen", rgb_annotated_image)


    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
