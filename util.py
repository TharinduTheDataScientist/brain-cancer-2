import base64
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.vgg16 import preprocess_input

def set_background(image_file):
   
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-position: center center;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_map):
    #convert image to (224,224)
    img_arr = np.asarray(image) #img to array
    img_arr = preprocess_input(img_arr) #use prepeocessing from vgg16
    img_arr = img_arr / 255.0 #rescale to 0-1
    img_arr = np.expand_dims(img_arr, axis=0) #expand dims from (224,224,3) -> (1,224,224,3)

    #make predictions
    prediction = model.predict(img_arr)
    result_idx = np.argmax(prediction)
    result = class_map[str(result_idx)]
    return result, prediction[0][result_idx]*100

