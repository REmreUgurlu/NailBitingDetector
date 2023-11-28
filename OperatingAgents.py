# def writer(time_interval, is_biting):
#     face_lms = []
#     hand_lms = []
#     cap_face = cv2.VideoCapture(0)
#     face_detector = FaceMeshDetector()
#     hand_detector = HandDetector()


#     success_capture, img_face = cap_face.read()
#     success_face, face_lms = face_detector.findFaceMesh(img_face)

#     landmarks = []

#     if success_face == False:
#         print("No Face Detected")
#         return
#     else:
#         success_cap, img_hand = cap_face.read()
#         success_hand, hand_lms, img = hand_detector.findHands(img_hand, True)
#         if success_hand == True:
#             for k in range(0,2):
#                 print("adding face lms")
#                 landmarks.append(face_lms[0][k])
#             for i in range(0,5):
#                 for j in range(0, 2):
#                     landmarks.append(hand_lms[i][j])
#             landmarks.append(is_biting)
#         else:
#             print("No Hands detected")
#             # t.cancel()
#             return

#     landmarks = np.array(landmarks)
#     landmarks_reshaped = landmarks.reshape((1,13))
#     writer_sample = data_access.Writer(landmarks_reshaped)
#     df = writer_sample.write_to_csv()
    
#     return(print(df))
    

# def reader():
#     reader = data_access.Reader()
#     train_dataset, validation_dataset, test_dataset = reader.read_with_parameters()

#     train_features = train_dataset.copy()
#     validation_features = validation_dataset.copy()
#     test_features = test_dataset.copy()

#     train_labels = train_features.pop('is_biting')
#     validation_labels = validation_features.pop('is_biting')
#     test_labels = test_features.pop('is_biting')

#     classification_model = classifier.Classify(train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

#     result = classification_model.classification(epochs=100)
    
# def predicter():
#     face_lms = []
#     hand_lms = []
#     cap_face = cv2.VideoCapture(0)
#     res, img_face = cap_face.read()
#     face_detector = FaceMeshDetector()  
#     hand_detector = HandDetector()

#     success_face, face_lms = face_detector.findFaceMesh(img_face)

#     landmarks = []

#     if success_face == False:
#         print("No Face Detected")
#         return
#     else:
#         success_cap, img_hand = cap_face.read()
#         success_hand, hand_lms, img = hand_detector.findHands(img_hand, True)
#         if success_hand == True:
#             for k in range(0,2):
#                 landmarks.append(face_lms[k])
#             for i in range(0,5):
#                 for j in range(0, 2):
#                     landmarks.append(hand_lms[i][j])
#         else:
#             print("No Hands detected")
#             return False, img

#     reader = data_access.Reader()
#     train_dataset, validation_dataset, test_dataset = reader.read_with_parameters()

#     train_features = train_dataset.copy()
#     validation_features = validation_dataset.copy()
#     test_features = test_dataset.copy()

#     train_labels = train_features.pop('is_biting')
#     validation_labels = validation_features.pop('is_biting')
#     test_labels = test_features.pop('is_biting')

#     prediction_model = classifier.Classify(train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

#     result = prediction_model.prediction(landmarks)
#     return result, img
