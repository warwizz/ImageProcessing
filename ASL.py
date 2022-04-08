import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from keras.preprocessing import image
from io import BytesIO
from urllib.request import urlopen
from PIL import Image

model = tf.keras.models.load_model("model_asl.h5")

label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
       'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def predict_image(image_upload, model = model):
    im = Image.open(image_upload)
    resized_im = im.resize((100, 100))
    im_array = np.asarray(resized_im)
    im_array = im_array*(1/255)
    im_input = tf.reshape(im_array, shape = [1, 100, 100, 3])


    classes = pd.DataFrame(model.predict(im_input,batch_size=1)).T
    class_labels = pd.DataFrame(label)
    res = pd.concat([class_labels,classes], axis=1)
    res.columns = ['Label','Pred']
    p = res.loc[res.Pred==res.Pred.max()]
    pred = p.iloc[0]
    return pred.Label, pred.Pred


st.title("Final Project Image Processing from Data Bangalore ")
st.header('Project by Wisnu Waskitho Putra')
st.markdown("""
                    """)
st.markdown("""
                    """)

st.subheader("Paste the link for identifying the ASL Language")
image_url = st.text_input("Please use JPG or JPEG image for better prediction")
st.write("Using Kaggle dataset provides better result because of the variance of the dataset")
try:
    if st.button("Classify the image"):
        file = BytesIO(urlopen(image_url).read())
        img = file
        label, pred = predict_image(img)
        st.header('The image is:')
        st.header(label)
        
except:
    st.markdown("<h1 style='text-align: center; color:red;'>Can't Open the Link! </h1>", unsafe_allow_html=True)
    st.markdown("""
                        """)
    col1, col2, col3 = st.columns([1,1.5,1])
    with col1:
        st.write("")

    with col2:
        st.image("no.png", width=300)

    with col3:
        st.write("")
    st.markdown("<h2 style='text-align: center;'>Please Use Another Link Image!</h2>", unsafe_allow_html=True)
