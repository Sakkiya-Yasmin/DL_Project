import streamlit as st
import pickle
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def names(number):
    if number == 1:
        return 'with helmet'
    else:
        return 'without helmet'

def main():
    st.markdown("<h1 style='text-align: center; color: #FFFFFF; background-color: #000000'>Helmet Detection App</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: left; color: #000000; background-color: #e3d5eb'> \nThis is a helmet detection application that can accurately identify individuals wearing helmets or not in images.</p>",
        unsafe_allow_html=True)
    st.subheader("How to Use:")
    st.markdown(
        "<p style='text-align: left; color: #000000; background-color: #e3d5eb'>1. Click the \"Browse files\" button and select an image from your computer.</p>",
        unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align: left; color: #000000; background-color: #e3d5eb'>2. Click the \"Predict\" button below and the image will be predicted.</p>",
        unsafe_allow_html=True)


    model = load_model('my_model.h5')
    st.write("Please upload an image to detect helmets")
    uploaded_file = st.file_uploader("Choose an image:", type=['jpg', 'jpeg'])

    if uploaded_file:
        img = Image.open(uploaded_file)
        img = img.resize((128, 128))
        img = np.array(img)
        if (img.shape == (128, 128, 3)):
            img = np.expand_dims(img, axis=0)
        if st.button("Predict"):
            prediction = model.predict_on_batch(img)
            classification = np.where(prediction == np.amax(prediction))[1][0]
            st.image(img, width=400)
            st.write(" This Is " + names(classification))


main()






