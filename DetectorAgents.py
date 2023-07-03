import DataAccess.DataAccess as data_access
import Classifier as classifier

import cv2
import mediapipe as mp
import numpy as np
import threading


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
            mouth_pos_x = round(mouth_pos_x, 6)
            mouth_pos_y = round(mouth_pos_y, 6)
            positions = [mouth_pos_x, mouth_pos_y]
            face.append(positions)
            return True, face
        else:
            return False, face

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

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return True, self.findPosition(img)
        else:
            return False, None

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                if id % 4 == 0 and id != 0:
                    rounded_x, rounded_y = round(lm.x,6), round(lm.y, 6) 
                    lmList.append([rounded_x, rounded_y])
                    # if draw:
                    #     cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)

        return lmList
    
def writer(time_interval, is_biting):
    face_lms = []
    hand_lms = []
    cap_face = cv2.VideoCapture(0)
    face_detector = FaceMeshDetector()
    hand_detector = HandDetector()


    success_capture, img_face = cap_face.read()
    success_face, face_lms = face_detector.findFaceMesh(img_face)

    t = threading.Timer(time_interval, writer, args=(time_interval, is_biting))
    t.start()
    landmarks = []

    if success_face == False:
        print("No Face Detected")
        t.cancel()
        return
    else:
        success_cap, img_hand = cap_face.read()
        success_hand, hand_lms = hand_detector.findHands(img_hand, True)
        if success_hand == True:
            for k in range(0,2):
                landmarks.append(face_lms[0][k])
            for i in range(0,5):
                for j in range(0, 2):
                    landmarks.append(hand_lms[i][j])
            landmarks.append(is_biting)
        else:
            print("No Hands detected")
            t.cancel()
            return

    landmarks = np.array(landmarks)
    landmarks_reshaped = landmarks.reshape((1,13))
    writer_sample = data_access.Writer(landmarks_reshaped)
    df = writer_sample.write_to_csv()
    
    return(print(df))
    

def reader():
    reader = data_access.Reader()
    train_dataset, validation_dataset, test_dataset = reader.read_with_parameters()

    train_features = train_dataset.copy()
    validation_features = validation_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('is_biting')
    validation_labels = validation_features.pop('is_biting')
    test_labels = test_features.pop('is_biting')

    classification_model = classifier.Classify(train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

    result = classification_model.classification(epochs=100)
    
def predicter():
    face_lms = []
    hand_lms = []
    cap_face = cv2.VideoCapture(0)
    res, img = cap_face.read()
    face_detector = FaceMeshDetector()  
    hand_detector = HandDetector()

    success_capture, img_face = cap_face.read()
    success_face, face_lms = face_detector.findFaceMesh(img_face)

    landmarks = []

    if success_face == False:
        print("No Face Detected")
        return
    else:
        success_cap, img_hand = cap_face.read()
        success_hand, hand_lms = hand_detector.findHands(img_hand, True)
        if success_hand == True:
            for k in range(0,2):
                landmarks.append(face_lms[0][k])
            for i in range(0,5):
                for j in range(0, 2):
                    landmarks.append(hand_lms[i][j])
        else:
            print("No Hands detected")
            return False, img_face

    reader = data_access.Reader()
    train_dataset, validation_dataset, test_dataset = reader.read_with_parameters()

    train_features = train_dataset.copy()
    validation_features = validation_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('is_biting')
    validation_labels = validation_features.pop('is_biting')
    test_labels = test_features.pop('is_biting')

    prediction_model = classifier.Classify(train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

    result = prediction_model.prediction(landmarks)
    return result, img_face


def main():
    # Use below code to add new rows to the dataset.csv
    # writer(time_interval=5, is_biting=False)

    # Below lines of code will try to predict if you are biting your nail
    result, img = predicter()
    print(np.argmax(result))
    cv2.imshow("img", img)
    cv2.waitKey(10000)
