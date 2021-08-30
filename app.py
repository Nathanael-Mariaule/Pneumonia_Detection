import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import cv2 as cv
import numpy as np
import os
import io
import PIL.Image as Image

os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
physical_devices = tf.config.list_physical_devices('CPU') 



@tf.function
def preprocessing(file, img_size=(150,150)):
    img = Image.open(io.BytesIO(file))
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img = cv.resize(img, img_size, interpolation = cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   
    img = img/255
    img = np.reshape(img, (1, img_size[0],img_size[1],1))
    return img



st.title('Pneumonia Prediction')

mariaunet = tf.keras.models.load_model('mariaunet')

def predict(filename):
    img = preprocessing(filename)
    preds = mariaunet.predict(img)[0][0]
    if preds>0.5:
        return f'Normal ({100-preds*100:1f}% of confidence to have Pneumonia)', img
    else:
        return f'Pneumonia - {100-preds*100:1f}% of confidence', img





uploaded_file = st.file_uploader("Choose an image ")
left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Predict:')
if pressed:
    if not uploaded_file:
        right_column.write('Please, upload an image')
    else:
        label, img = predict(uploaded_file.getvalue())
        right_column.write(f'{label}')
        st.image(uploaded_file.getvalue())

