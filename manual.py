import re
import os
import cv2
import keras
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras import metrics
from keras.models import load_model

def top_3_acc(y_true, y_pred):
	return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def load_image(image_location):
	img = cv2.imread(image_location)
        
	img = cv2.resize(img, (28, 28))
 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); img = img[..., None]

	img = np.array(img)

	img = img.astype('float16') / 255.
    
	img = tf.expand_dims(img, axis=0)
     
	return img

def classifyDrawing(image_location):   
	img = load_image(image_location)
    
	model = load_model(os.getcwd() + "/model/model.h5", custom_objects={"top_3_acc": top_3_acc})

	csv = pd.read_csv(os.getcwd() + "/data.csv")
 
	array = csv.values.tolist()
  
	data = []

	for _ in array:
		data.append(_[1])

	pred = model.predict(img, steps=1)

	answer = np.argmax(pred)

	print("Prediction: ", data[answer-1])

	return True

classifyDrawing("./images/1.png")