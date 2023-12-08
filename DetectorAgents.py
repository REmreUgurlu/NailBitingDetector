import DataAccess as data_access
import Classifier as classifier
import cv2
import mediapipe as mp
import numpy as np
import time


class FaceMeshDetector():
    
    def __init__(self,staticMode=False, maxFaces=1, refineLms=True, minDetectionCon = 0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLms = refineLms
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon   

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLms, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        face = []
        if self.results.multi_face_landmarks:
            self.my_face = self.results.multi_face_landmarks[0]
            mouth_pos_x = self.my_face.landmark[13].x
            mouth_pos_y = self.my_face.landmark[13].y
            mouth_pos_x = round(mouth_pos_x, 3)
            mouth_pos_y = round(mouth_pos_y, 3)
            # print(mouth_pos_x,  mouth_pos_y)
            ih,iw,_ = img.shape
            positions = [mouth_pos_x, mouth_pos_y]
            drawing_positions = [int(mouth_pos_x * iw), int(mouth_pos_y * ih)]
            # print(drawing_positions)
            cv2.circle(img, drawing_positions, 5, (255,0,0), -1)

            return True, positions
        else:
            return False, face

class FaceDetector():
    def __init__(self, model_selection=1, min_detec_conf=0.5):
        self.model_selection = model_selection
        self.min_detec_conf = min_detec_conf

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(self.model_selection, self.min_detec_conf)

    def find_mouth(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(imgRGB)
        if results.detections:
            for detection in results.detections:
                landmarks = detection.location_data.relative_keypoints

                ih,iw,_ = img.shape
                mouth = (int(landmarks[3].x * iw), int(landmarks[3].y * ih))

                cv2.circle(img,mouth, 5, (0,255,0), -1)
            return True

        # mouth_pos = self.mp_face_detection.get_key_point()
        

        

class HandDetector():
    def __init__(self, mode=False, maxHands=1, modComplex=1, detectionCon=0.2, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modComplex = modComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        is_successfull = False
        is_biting = False
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            cv2.imshow('frame1', img)
            if cv2.waitKey(2000) == ord('q'):
                # is_successfull = False
                return is_successfull, None, imgRGB, is_biting
            elif cv2.waitKey(4000) == ord('w'):
                is_successfull = True
                is_biting = True
            else:
                is_successfull = True
                pass
            cv2.destroyWindow('frame1')
            time.sleep(1)
            lm_list = self.findPosition()
            return is_successfull, lm_list, img, is_biting
        else:
            return is_successfull, None, imgRGB, is_biting

    def findPosition(self, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                if id % 4 == 0 and id != 0:
                    rounded_x, rounded_y = round(lm.x,6), round(lm.y, 6) 
                    lmList.append(rounded_x)
                    lmList.append(rounded_y)


        return lmList


def main():
    face_mesh = FaceMeshDetector()
    hand_mesh = HandDetector()
    face_lms = []
    hand_lms = []
    lms = []
    cap_face = cv2.VideoCapture(0)

    success_capture, img_face = cap_face.read()
    success_face, face_lms = face_mesh.findFaceMesh(img_face)
    if success_face == False:
        print("No Face Detected")
        return
    else:
        lms = face_lms.copy()
        success_hand, hand_lms, img, is_biting = hand_mesh.findHands(img_face, True)
        if success_hand == True:
            face_lms.extend(hand_lms)
            # is_biting 
            # cv2.imshow("frame2", img)
            # if cv2.waitKey(2000) == ord('e'):
            #     is_biting = False
            # cv2.destroyWindow('frame2')
            # elif cv2.waitKey(2500) == ord('q'):
            #     return
            for i in range(0,10,2):
                pos_x = round(lms[0] - hand_lms[i],6)            
                pos_y = round(lms[1] - hand_lms[i+1],6)
                positions = [pos_x, pos_y]
                lms.extend(positions)
            lms.append(is_biting)
            final_lms = lms[2:]
            return print(final_lms)
        else:
            print("No Hands detected")
            return

if __name__ == "__main__":
    main()
