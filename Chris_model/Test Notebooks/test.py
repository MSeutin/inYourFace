import face_recognition
import pickle
import os
import cv2

LABELED_FACES_DIR = "../Dataset/Training/"
UNLABELED_FACES_DIR = "../Dataset/Validation"
TEST_DIR = "../Dataset/Testing/"
TOLERANCE = 0.55
FRAME_THICKNESS = 2
FONT_THICKNESS = 1

with open('known_faces.dat','rb') as f:
    known_faces = pickle.load(f)

with open('known_names.dat','rb') as f:
    known_names = pickle.load(f)

print(known_names)


for filename in os.listdir(UNLABELED_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNLABELED_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model='cnn')
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            #A list of tuples of found face locations in css (top, right, bottom, left) order
            top_left = (face_location[3]-10, face_location[0]-50)
            bottom_right = (face_location[1]+10, face_location[2]+50)
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3] - 11, face_location[2] + 50)
            bottom_right = (face_location[1] + 11, face_location[2] + 72)
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3], face_location[2]+65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), FONT_THICKNESS)

    cv2.imshow(filename, image)
    cv2.waitKey(1000)










