import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import load_img,img_to_array

names = ["Anudari", "Cebastian", "Chris", "Michael"]
mask_label = ['with mask', 'without mask']


####################
### MASK/NO MASK ###
####################
mask_no_mask_model = tf.keras.models.load_model('./MODELS/chris_mask_nomask_model.h5')
#mask_no_mask_model.summary()


#####################
### UNMASKED FACE ###
#####################
unmasked_model = tf.keras.models.load_model('./MODELS/new_nomask_model.h5')
#unmasked_model.summary()


###################
### MASKED FACE ###
###################
masked_model = tf.keras.models.load_model('./MODELS/new_mask_model.h5')
#masked_model.summary()


######################
### TEST ONE IMAGE ###
######################
'''
my_image = load_img('../NEW_DATASET/masked/michael/IMG_6086.JPG', target_size=(224, 224))

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
'''

##################################
### TEST MULTIPLE MIXED IMAGES ###
##################################
'''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_data_dir = '../NEW_DATASET/mixed/'
test_datagen = ImageDataGenerator()
Test_data = test_datagen.flow_from_directory(
                        test_data_dir,
                        batch_size = 1,
                        target_size=(224,224),
                        shuffle = False) 

#Predict test images
Y_pred = mask_no_mask_model.predict(Test_data) #fixme:: branch to which ever is mask or no mask

#Get corresponding predicted label
y_pred = np.argmax(Y_pred, axis=1)

cf = confusion_matrix(Test_data.classes, y_pred)


df_cm = pd.DataFrame(cf, index=names, columns=names)

sns.heatmap(df_cm, annot= True,fmt="d",cmap="YlGnBu")
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("Confusion matrix\n")
plt.show()
'''

######################
### VIDEO WITH DNN ###
######################
predictionLabel = ''
boxColor = (0,0,0)

network = cv2.dnn.readNetFromCaffe("./MODELS/deploy.prototxt", "./MODELS/res10_300x300_ssd_iter_140000_fp16.caffemodel")

video = cv2.VideoCapture(0)
#video = cv2.VideoCapture("../Dataset/Video1.mp4")


if (video.isOpened() == False):
    print("Web Camera not detected")
while (True):
    ret, frame = video.read()
    if ret == True:
        ### DETECT FACE LOCATION USING DNN
        dnn_label = ''
        dnn_boxColor = (0,0,0)

        dnn_image = cv2.resize(frame, (582,448))

        (height, width) = dnn_image.shape[:2]
        blob = cv2.dnn.blobFromImage(dnn_image, scalefactor=1.0, size=(300, 300), mean=(104.0, 117.0, 123.0))
        network.setInput(blob)
        detections = network.forward()
        

        ###
        new_frame = cv2.resize(dnn_image, (224,224))
        x = image.img_to_array(new_frame)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        dnn_prediction = mask_no_mask_model.predict(x)

        if mask_label[np.argmax(dnn_prediction, axis=1)[0]] == 'with mask':
            dnn_label = 'Masked - '
            dnn_prediction = masked_model.predict(x)
            dnn_boxColor = (255,0,0)
        else:
            dnn_label = 'Unmasked - '
            dnn_prediction = unmasked_model.predict(x)
            dnn_boxColor = (0,0,255)

        dnn_label = dnn_label + names[np.argmax(dnn_prediction, axis=1)[0]]
        ###

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                left = int(box[0])
                top = int(box[1])
                right = int(box[2])
                bottom = int(box[3])

                cv2.rectangle(dnn_image, (left - 10, top - 50), (right + 10, bottom + 50), dnn_boxColor, 2)
                cv2.rectangle(dnn_image, (left - 11, top - 49), (right + 11, top -24), dnn_boxColor, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(dnn_image, dnn_label, (left - 6, top - 36), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("DNN", dnn_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()