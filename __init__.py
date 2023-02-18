import tensorflow as tf
import cv2
import numpy as np
model_name = "UEH_vending_3class_20epoch_max.h5"
model = tf.keras.models.load_model(model_name)

cap = cv2.VideoCapture(1)
list_names = ['Alu', 'Milk_box', 'PET']
while True:
    _, frame = cap.read()
    image = cv2.resize(frame.copy(),(224,224))
    image = np.asarray([image])
    predict = model.predict(image)
    id = np.argmax(predict[0])
    print(list_names[id] + ":" + str(predict[0][id]))
    cv2.imshow("image", frame)
    cv2.waitKey(1)
