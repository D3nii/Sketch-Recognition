import re
import io
import os
import cv2
import keras
import base64
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras import metrics
from keras.models import load_model
from matplotlib import pyplot as plt

def _base64_to_image(base_string):
    base_string = base64.b64decode(base_string)
    
    base_string = io.BytesIO(base_string)
    
    base_string = Image.open(base_string)
    
    base_string = cv2.cvtColor(np.array(base_string), cv2.COLOR_RGB2BGR)
        
    base_string = cv2.resize(base_string, (28, 28))
 
    base_string = cv2.cvtColor(base_string, cv2.COLOR_BGR2GRAY)
    
    base_string = base_string[..., None]
    
    base_string = np.array(base_string)

    base_string = base_string.astype('float16') / 255.
    
    base_string = tf.expand_dims(base_string, axis=0)
    
    return base_string


def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def classifyDrawing(img):    
	img = re.sub('^data:image/.+;base64,', '', img)
    
	img = _base64_to_image(img)
	
	model = load_model(os.getcwd() + "/model/model.h5", custom_objects={"top_3_acc": top_3_acc})

	csv = pd.read_csv(os.getcwd() + "/data.csv")
 
	array = csv.values.tolist()
  
	data = []

	for _ in array:
		data.append(_[1])

	pred = model.predict(img, steps=32)

	answer = np.argmax(pred)

	print("Prediction: ", data[answer-1])

	return data[answer-1]
