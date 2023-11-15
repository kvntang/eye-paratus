#!/usr/bin/env python3
'''Animates distances and measurment quality'''
from rplidar import RPLidar
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

try:
    print("Waiting for Arduino to initialize...")

    while True:

        if not isArduinoReady:
            while ser.in_waiting > 0:
                data_received = ser.readline().decode().strip()
                print(f"Rcv >> {data_received}")
                isArduinoReady = True
                lidar = RPLidar(PORT_NAME)
            continue

        for i, scan in enumerate(lidar.iter_scans()):   # this is a forever loop

            angles = np.array([meas[1] for meas in scan])
            distances = np.array([meas[2] for meas in scan])

            if len(distances) > 0:
                min_idx = np.argmin(distances)
            else:
                min_idx = None

            # print(f"Closest Angle: {angles[min_idx]}, Distance: {distances[min_idx]}")
            closestAngle = round(angles[min_idx])

            currTime = time.time()
            if (currTime - prevTime > updateFreq):
                # print(f"{time.time()}: {closestAngle}")
                print("----------")
                print(f"Closest angle: {closestAngle}")
                # print("internal rotation before adding: ", internalRotation)
                prevTime = currTime

                if closestAngle > 3 and closestAngle < 357:
                    if closestAngle < 180:      # clockwise condition
                        movementValue = closestAngle
                    else:   # counter-clockwise condition
                        movementValue = closestAngle - 360
                    print("movement value: ", movementValue)
                    binary_data_to_send = struct.pack("iiii", 1, movementValue, 0, 0)
                    ser.write(binary_data_to_send)

except KeyboardInterrupt:
    lidar.stop()
    lidar.disconnect()
    ser.close()  # Close the serial connection on keyboard interrupt
