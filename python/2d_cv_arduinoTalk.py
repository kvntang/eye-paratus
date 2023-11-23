#!/usr/bin/env python3

# from rplidar import RPLidar
### Note: pip install rplidar-roboticia
# import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.animation as animation

import serial
import time
import struct

PORT_NAME = 'COM7'
DMAX = 4000
IMIN = 0
IMAX = 50

isArduinoReady = False
arduinoPort = 'COM6'
# Establish the serial connection
ser = serial.Serial(arduinoPort, 115200)  # Use the appropriate port and baud rate
print(f"Arduino connected at port {arduinoPort}")
# lidar = RPLidar(PORT_NAME)

updateFreq = 0.2
prevTime = 0

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# import numpy as np

# For webcam input:
cap = cv2.VideoCapture(0)

success, image = cap.read()
if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    cap.release()
    exit()

height, width = image.shape[:2]
x_scale_factor = 1.62  # Modify this for different sizes
y_scale_factor = 2.15 
new_width = int(width * x_scale_factor)
new_height = int(height * y_scale_factor)

y_translation = -33  # Change this value as needed
translation_matrix = np.float32([[1, 0, 0], [0, 1, y_translation]])
# x_translation = 50
# translation_matrix = np.float32([[1, 0, x_translation], [0, 1, 0]])

# Create a window
cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)

# Set window to fullscreen
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

isRun = True

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

        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # blacking out here
                image = np.zeros(image.shape, np.uint8)

                pos = None

                # Draw just the thumb and print position
                lmList = []
                if results.multi_hand_landmarks:
                    myhand = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myhand.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y*h)
                        lmList.append([id, cx, cy])
                    if len(lmList) != 0:
                        pos = lmList[8] #show only one landmark
                        cv2.circle(image, (pos[1], pos[2]), 15, (0, 255, 0), cv2.FILLED)
                        text = f'{str(pos[1])}, {str(pos[2])} '
                        cv2.putText(image, text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                        # draw line between point and center of screen
                        # cv2.line(image, (pos[1], pos[2]), (300, 225), (0, 255, 0), thickness=2, lineType=8)
                
                resized_frame = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                x_center = new_width // 2
                y_center = new_height // 2

                startx = x_center - (width // 2)
                starty = y_center - (height // 2)
                cropped_frame = resized_frame[starty:starty+height, startx:startx+width]
                
                translated_frame = cv2.warpAffine(cropped_frame, translation_matrix, (cropped_frame.shape[1], cropped_frame.shape[0]))

                # cv2.imshow("Camera Feed", cropped_frame)
                cv2.imshow("Camera Feed", translated_frame)
                # cv2.imshow("Camera Feed", image)

                currTime = time.time()
                if (currTime - prevTime > updateFreq):
                    # print("----------")
                    # print(f"Closest angle: {closestAngle}")
                    prevTime = currTime

                    try:
                        moveX = pos[1]-width/2
                        moveY = pos[2]-height/2-20
                        print(f"x: {moveX}, y: {moveY}")

                        # if moveX > 5 or moveX < -5 or moveY > 5 or moveY < -5:
                        if abs(moveX) > 100:
                            # when x is negative, move counter-clockwise
                            if moveX < 0:
                                movementValue = -5
                            # when x is positive, move clockwise
                            else: 
                                movementValue = 5
                            print("movement value: ", movementValue)
                            # 0 is absolute, 1 is relative
                            binary_data_to_send = struct.pack("iiii", 1, movementValue, 0, 0)
                            ser.write(binary_data_to_send)
                    except:
                        print("No hands detected")

                if cv2.waitKey(5) & 0xFF == 27:
                    isRun = False
                    ser.close()
                    break

        # for i, scan in enumerate(lidar.iter_scans()):   # this is a forever loop

        #     angles = np.array([meas[1] for meas in scan])
        #     distances = np.array([meas[2] for meas in scan])

        #     if len(distances) > 0:
        #         min_idx = np.argmin(distances)
        #     else:
        #         min_idx = None

        #     # print(f"Closest Angle: {angles[min_idx]}, Distance: {distances[min_idx]}")
        #     closestAngle = round(angles[min_idx])

        #     currTime = time.time()
        #     if (currTime - prevTime > updateFreq):
        #         # print(f"{time.time()}: {closestAngle}")
        #         print("----------")
        #         print(f"Closest angle: {closestAngle}")
        #         # print("internal rotation before adding: ", internalRotation)
        #         prevTime = currTime

        #         if closestAngle > 3 and closestAngle < 357:
        #             if closestAngle < 180:      # clockwise condition
        #                 movementValue = closestAngle
        #             else:   # counter-clockwise condition
        #                 movementValue = closestAngle - 360
        #             print("movement value: ", movementValue)
        #             # 0 is absolute, 1 is relative
        #             binary_data_to_send = struct.pack("iiii", 1, movementValue, 0, 0)
        #             ser.write(binary_data_to_send)

except KeyboardInterrupt:
    # lidar.stop()
    # lidar.disconnect()
    ser.close()  # Close the serial connection on keyboard interrupt

cap.release()
cv2.destroyAllWindows