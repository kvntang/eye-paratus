import serial
import time
import struct

isArduinoReady = False
arduinoPort = 'COM6'

# Establish the serial connection
ser = serial.Serial(arduinoPort, 115200)  # Use the appropriate port and baud rate
print(f"Arduino connected at port {arduinoPort}")

try:
  print("Waiting for Arduino to initialize...")

  ### For debugging Arduino
  # while True:
  #   if ser.in_waiting > 0:
  #     data_received = ser.read(ser.in_waiting).decode().strip()
  #     print(f"Rcv >> {data_received}")

  while True:

    if not isArduinoReady:
      while ser.in_waiting > 0:
        data_received = ser.readline().decode().strip()
        print(f"Rcv >> {data_received}")
        isArduinoReady = True
      continue

    try:
      # Read data from the serial machine
      while ser.in_waiting > 0:
        data_received = ser.readline().decode().strip()
        print(f"Rcv >> {data_received}")
      # Send data to the serial machine
      user_input = input("Send >> ")
      input_list = user_input.split()
      # ser.write(data_to_send.encode())
      binary_data_to_send = struct.pack("iiii", 0, int(input_list[0]), int(input_list[1]), int(input_list[2]))
      ser.write(binary_data_to_send)
    except ValueError as e:
      print(f"Error: {e}")
    except Exception as e:
      print(f"An error occurred: {e}")

    # Wait for a short time to let the serial machine process the data
    time.sleep(0.1)

    # Read data from the serial machine
    # if ser.in_waiting > 0:
    #   data_received = ser.read(ser.in_waiting).decode().strip()
    #   print(f"Rcv >> {data_received}")

except KeyboardInterrupt:
  ser.close()  # Close the serial connection on keyboard interrupt
