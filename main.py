import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify, set_background
import json

#set background
set_background('background.jpg')

#set tittle
st.title("Brain Cancer Classifier")

# set header
st.header("Please upload image of brain x-ray")

# upload file
file = st.file_uploader('image', type=['jpeg', 'webp', 'png', 'jpg'])

# load classifier
model = load_model('./model/brain_cancer_vgg16.h5')

# load class names
with open('./model/label_mapper.json') as f:
    class_mapper = json.load(f)

# display image & classify
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    image = image.resize((224,224), resample=Image.Resampling.LANCZOS)
    
    # classify image
    class_name, confidence = classify(image, model, class_mapper)

    # write classification
    st.write("## Brain cancer result: {}".format(class_name))
    st.write("## Confidence: {}%".format(round(confidence,4)))
    #st.write("### AI is {:.2f}% confident".format(conf_score*100))                                                              
