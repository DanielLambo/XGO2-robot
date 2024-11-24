# Import necessary libraries
import cv2
import os, socket, sys, time
import spidev as SPI
import xgoscreen.LCD_2inch as LCD_2inch
from PIL import Image, ImageDraw, ImageFont
from key import Button
import numpy as np
import mediapipe as mp
from numpy import linalg
from xgolib import XGO

# Initialize XGO robot connection
dog = XGO(port='/dev/ttyAMA0',version="xgolite")
dogtime = 0

# Initialize LCD display
display = LCD_2inch.LCD_2inch()
display.clear()
splash = Image.new("RGB", (display.height, display.width ),"black")
display.ShowImage(splash)
button=Button()

# ----------------------- COMMON INIT ------------------------

# Import libraries for hand detection
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Function to detect if a finger is stretched based on landmark points
def finger_stretch_detect(point1, point2, point3):
  result = 0
  dist1 = np.linalg.norm((point2 - point1), ord=2)  # Calculate distance between points
  dist2 = np.linalg.norm((point3 - point1), ord=2)
  if dist2 > dist1:
    result = 1  # Finger is stretched if point3 is further than point2
  return result

# Function to detect hand gesture based on finger stretch
def detect_hands_gesture(result):
  gesture = ""
  # Define logic for different finger combinations (replace with your desired gestures)
  if (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
    gesture = "good"
  elif (result[0] == 0) and (result[1] == 1)and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
    gesture = "one"
  # ... Add more gesture definitions here ...
  elif(result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
    gesture = "stone"
  else:
    gesture = "not in detect range..."
  
  return gesture

# For static images (not used in this example)
IMAGE_FILES = []
# ... (code for processing static images omitted) ...

# For webcam input
cap=cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))

figure = np.zeros(5)  # Array to store finger stretch results
landmark = np.empty((21, 2))  # Array to store landmark coordinates

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can not receive frame (stream end?). Exiting...")
        break
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_RGB)
    frame_height = frame.shape[0]
    frame_width  = frame.shape[1]
    gesture_result=[]
    if result.multi_hand_landmarks:
        for i, handLms in enumerate(result.multi_hand_landmarks):
            mpDraw.draw_landmarks(frame, 
                                  handLms, 
                                  mpHands.HAND_CONNECTIONS,
              mpDraw.draw_landmarks(frame, 
                                  handLms, 
                                  mpHands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=handLmsStyle,
                                  connection_drawing_spec=handConStyle)   


            # Extract landmark coordinates
            for j, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * frame_width)
                yPos = int(lm.y * frame_height)
                landmark_ = [xPos, yPos]
                landmark[j,:] = landmark_   


            # Calculate finger stretch for each finger
            for k in range (5):
                if k == 0:
                    figure_ = finger_stretch_detect(landmark[17],landmark[4*k+2],landmark[4*k+4])
                else:    
                    figure_ = finger_stretch_detect(landmark[0],landmark[4*k+2],landmark[4*k+4])

                figure[k] = figure_   


            # Detect hand gesture based on finger stretch results
            gesture_result = detect_hands_gesture(figure)

    # Display gesture on the frame
    b,g,r = cv2.split(frame)
    frame = cv2.merge((r,g,b))
    frame = cv2.flip(frame, 1)
    if result.multi_hand_landmarks:
      cv2.putText(frame, f"{gesture_result}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255 ,255,   
 0), 5)

    # Control the robot based on the detected gesture
    if time.time()>dogtime:
      if gesture_result=="good":
        dogtime=time.time()
        dog.action(23)
        dogtime+=3
      elif gesture_result=="one":
        # ... Add more gesture-action mappings here ...
      elif gesture_result=="stone":
        dogtime=time.time()
        dog.action(20)
        dogtime+=3

    # Display the frame on the LCD
    imgok = Image.fromarray(frame)
    display.ShowImage(imgok)

    # Check for exit conditions (keyboard interrupt or button press)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    if button.press_b():
      dog.reset()
      break

cap.release()
