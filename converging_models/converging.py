import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
#grab preprocess from pre-trained model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import load_img,img_to_array

from keras.models import load_model
from keras_vggface import utils

names = [
        'Angelina_Jolie', 'Brad_Pitt', 'Denzel_Washington', 'Hugh_Jackman', 'Jennifer_Lawrence', 'Johnny_Depp', 
        'Kate_Winslet', 'Leonardo_DiCaprio', 'Megan_Fox', 'Natalie_Portman', 'Nicole_Kidman',
        'Robert_Downey_Jr', 'Sandra_Bullock', 'Scarlett_Johansson', 'Tom_Cruise','Tom_Hanks', 'Will_Smith'
        ]

mask_label = ['with mask', 'without mask']

####################
### MASK/NO MASK ###
####################
from tensorflow.keras.applications.vgg19 import preprocess_input

mask_no_mask_model = tf.keras.models.load_model('./MODELS/chris_mask_nomask_model.h5')
mask_no_mask_model.summary()

#####################
### UNMASKED FACE ###
#####################
unmasked_model = tf.keras.models.load_model('./MODELS/michael_no_mask_model.h5')
unmasked_model.summary()

###################
### MASKED FACE ###
###################
from tensorflow.keras.applications.vgg19 import preprocess_input

masked_model = tf.keras.models.load_model('./MODELS/cebastion_mask_model.h5')
masked_model.summary()

######################
### TEST ONE IMAGE ###
######################
#my_image = load_img('../IN_YOUR_FACE_DATASET/MASKED_CELEBRITIES/Megan_Fox/Megan_Fox_006_cloth.jpg', target_size=(224, 224))
my_image = load_img('../converging_models/converge_testing/test/Sandra_Bullock_098_KN95.jpg', target_size=(224, 224))

my_image = img_to_array(my_image)
my_image = np.expand_dims(my_image, axis=0)
my_image = preprocess_input(my_image)

predictionLabel= ''
prediction = mask_no_mask_model.predict(my_image)

if mask_label[np.argmax(prediction, axis=1)[0]] == 'with mask':
    predictionLabel = 'Masked - '
    prediction = masked_model.predict(my_image)
else:
    predictionLabel = 'Unmasked - '
    prediction = unmasked_model.predict(my_image)

predictionLabel = predictionLabel + names[np.argmax(prediction, axis=1)[0]]


print(predictionLabel)

#############
### MTCNN ###
#############
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

predictionLabel = ''
boxColor = (0,0,0)

video = cv2.VideoCapture(0)
#video = cv2.VideoCapture("../Dataset/Video1.mp4")


if (video.isOpened() == False):
    print("Web Camera not detected")
while (True):
    ret, frame = video.read()
    if ret == True:
        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x_point, y_point, width, height = face['box']
                x2_point, y2_point = x_point + width, y_point + height

                ###
                new_frame = cv2.resize(frame, (224,224))
                x = image.img_to_array(new_frame)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                prediction = mask_no_mask_model.predict(x)

                if mask_label[np.argmax(prediction, axis=1)[0]] == 'with mask':
                    predictionLabel = 'Masked - '
                    prediction = masked_model.predict(x)
                    boxColor = (255,0,0)
                else:
                    predictionLabel = 'Unmasked - '
                    prediction = unmasked_model.predict(x)
                    boxColor = (0,0,255)

                predictionLabel = predictionLabel + names[np.argmax(prediction, axis=1)[0]]
                ###
                cv2.rectangle(frame, (x_point - 10, y_point - 50), (x2_point + 10, y2_point + 50), boxColor, 2)
                cv2.rectangle(frame, (x_point - 11, y_point - 49), (x2_point + 11, y_point - 24), boxColor, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, predictionLabel, (x_point - 6, y_point - 36), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("Output",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()

########################
### FACE_RECOGNITION ###
########################
'''
import face_recognition

video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture("../Dataset/Video1.mp4")

# Initialize some variables
face_locations = []
process_this_frame = True
predictionLabel = ''
boxColor = (0,0,0)

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        new_frame = cv2.resize(frame, (224,224))
        x = image.img_to_array(new_frame)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction = mask_no_mask_model.predict(x)

        if mask_label[np.argmax(prediction, axis=1)[0]] == 'with mask':
            predictionLabel = 'Masked - '
            prediction = masked_model.predict(x)
            boxColor = (255,0,0)
        else:
            predictionLabel = 'Unmasked - '
            prediction = unmasked_model.predict(x)
            boxColor = (0,0,255)

        predictionLabel = predictionLabel + names[np.argmax(prediction, axis=1)[0]]
        ###

        

    #process_this_frame = not process_this_frame


    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left - 10, top - 50), (right + 10, bottom + 50), boxColor, 2)
        cv2.rectangle(frame, (left - 11, top - 49), (right + 11, top -24), boxColor, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, predictionLabel, (left - 6, top - 36), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
'''