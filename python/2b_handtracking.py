#MachineGaze Demo 2
#Handtracking
#1. detect hand position
#2. move rotation motor
#3. move tilt motor so hand is in mid range (20% - 80% height) of screen

import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxhands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxhands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmark:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo =0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmark[handNo]

            for id, lm in enumerate(myhand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y*h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                
        return lmList

def main():
    pTime = 0
    cTime = 0
    vid = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = vid.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4]) #print postion of only landmark number 4

        cTime = time.time()
        fps = 1 /(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()