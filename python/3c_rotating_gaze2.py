import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

import serial
import struct
import random


def visualize(image, detection_result) -> np.ndarray:
  for detection in detection_result.detections:
    if detection.categories[0].score < 0.2:
       continue

    # Draw bounding_box
    bbox = detection.bounding_box
    # x_scale = 400/320
    # y_scale = 224/320
    x_scale = 380/320
    y_scale = 204/320
    # scaled_bbox_origin_x = round(bbox.origin_x * x_scale) 
    # scaled_bbox_origin_y = round(bbox.origin_y * y_scale) 
    scaled_bbox_origin_x = round(bbox.origin_x * x_scale) + 10
    scaled_bbox_origin_y = round(bbox.origin_y * y_scale) + 10
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

def move_to(val,movementValue):
    rotation, tilt, focus = movementValue
    # print("RTF: ", rotation, tilt, focus)
    # 0 is absolute, 1 is relative
    binary_data_to_send = struct.pack("iiii", val, rotation, tilt, focus)
    ser.write(binary_data_to_send)

def get_bounding_box_mid_point(highest_detection):
    x_scale = 380/320
    y_scale = 204/320

    bbox = highest_detection.bounding_box

    scaled_bbox_origin_x = round(bbox.origin_x * x_scale) + 10
    scaled_bbox_origin_y = round(bbox.origin_y * y_scale) + 10
    scaled_bbox_width = round(bbox.width * x_scale)
    scaled_bbox_height = round(bbox.height * y_scale)
    start_point = scaled_bbox_origin_x, scaled_bbox_origin_y
    end_point = scaled_bbox_origin_x + scaled_bbox_width, scaled_bbox_origin_y + scaled_bbox_height
    mid_point = (start_point[0] + end_point[0])/2, (start_point[1] + end_point[1])/2
    # print(mid_point)

    return mid_point

def generate_random_coordinate(r_lower_range, r_upper_range, t_lower_range, t_upper_range):
    # rotation
    rotation_diff = random.randint(-20, 20)
    internal_rotation = internal_rotation + rotation_diff
    rotation = internal_rotation

    # tilt
    tilt_diff = random.randint(-5, 5)
    internal_tilt = internal_tilt + tilt_diff
    if internal_tilt < 0: #out of bound check
        internal_tilt = 0
    elif internal_tilt > 20:
        internal_tilt = 20
    tilt = internal_tilt

    # focus
    focus = 0
    
    new_coordinate = [rotation, tilt, focus]

    return new_coordinate

##########################################################################################################
##########################################################################################################

PORT_NAME = 'COM7'
DMAX = 4000
IMIN = 0
IMAX = 50
isArduinoReady = False
arduinoPort = 'COM6'
ser = serial.Serial(arduinoPort, 115200)  # Use the appropriate port and baud rate
print(f"Arduino connected at port {arduinoPort}")
# updateFreq = 0.2
# prevTime = 0

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

frame_index = 0

#Motor
isRun = True
move_to_new_coord = True
internal_rotation = 0
internal_tilt = 10

scene_num = 0
move_time = 0

#Main Motor Loop
try:
    print("Waiting for Arduino to initialize...")
    while isRun:
        if not isArduinoReady:
            while ser.in_waiting > 0:
                data_received = ser.readline().decode().strip()
                print(f"Rcv >> {data_received}")
                isArduinoReady = True
                # lidar = RPLidar(PORT_NAME)
            continue

        #Main Video Loop
        while cap.isOpened():

            #clock setup
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
            opencv_frame = opencv_frame[140:364,123:523] # custom resizing, match with projector specs
            detection_frame = opencv_frame[10:214,10:390]
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB) #BGR to RGB
            resized_frame = cv2.resize(rgb_frame, (320, 320))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_frame)
            frame_timestamp_ms = int(1000 * frame_index / video_file_fps)
            

            #boredom timer
            if scene_num == 0:
                if curr_time - move_time > 10:
                    move_to_new_coord = True
                    move_time = curr_time
            
            # 1. random coord
            if (move_to_new_coord): #bored
                #random coord
                new_coordinate = generate_random_coordinate(r_lower_range, r_upper_range, t_lower_range, t_upper_range)

                move_to(0, new_coordinate) #0 = absolute, 1 = relative
                move_to_new_coord = False
                move_time = curr_time
            
            #Detection
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            # 2. Optimize coord to Object
            if curr_time - move_time > 0.5:     # update rate
                highest_score = 0
                highest_idx = -1
                # get highest detection
                for idx, detection in enumerate(detection_result.detections):
                    if highest_score < detection.categories[0].score:
                        highest_score = detection.categories[0].score
                        highest_idx = idx
                highest_detection = detection_result.detections[highest_idx]
                # print("highest detection: ", highest_detection.categories[0].score, highest_detection.categories[0].category_name)
                
                if highest_detection.categories[0].score > 0.3: #set threshold
                    # calculate center diff
                    mid_point = get_bounding_box_mid_point(highest_detection)
                    center_diff = mid_point[0] - 200, mid_point[1] - 112
                    # print(center_diff)

                    internal_rotation = round(internal_rotation + (center_diff[0]/40))
                    internal_tilt = round(internal_tilt + (center_diff[0]/40)) #size of movement
                    if internal_tilt < 0:
                        internal_tilt = 0
                    elif internal_tilt > 20:
                        internal_tilt = 20

                    # move
                    focus = 0
                    new_coordinate = [internal_rotation, internal_tilt, focus]
                    print(new_coordinate)
                    move_to(0, new_coordinate)
                    move_time = curr_time
            

            # image_copy = np.copy(mp_image.numpy_view())     # camera feed
            image_copy = np.full(opencv_frame.shape, 255, dtype=np.uint8)   # white
            annotated_image = visualize(image_copy, detection_result)
            rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("fullscreen", rgb_annotated_image)


            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                isRun = False
                ser.close()
                break

            frame_index += 1

except KeyboardInterrupt:
    # lidar.stop()
    # lidar.disconnect()
    ser.close()  # Close the serial connection on keyboard interrupt


cap.release()
cv2.destroyAllWindows()
