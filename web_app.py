import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from io import BytesIO

# Sidebar - The Team
st.sidebar.title("Sat Team")
st.sidebar.write("Rajat Sehgal")
st.sidebar.write("Francisco Varela Cid")
st.sidebar.write("Tofunmi Oludare")
st.sidebar.write("Shamil Aliyev")

@st.cache
# Load Image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Initial Screen + Upload image
def main():
    st.title("SATELLITE IMAGE CLASSIFIER")
    st.text("Here you can upload a satellite image and we will identify the objects in it.")

    uploaded_file = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        st.image(load_image(uploaded_file))

        if st.button("Click to Find Objects"):
            
            st.write(" ")
            st.write("------------")
            st.write("Check you image annotated below...")
            with st.spinner('Wait for it...'):
                # FUNCTION to run model on the uploaded_file
                # Returns the image that will be used below
                st.image(load_image(uploaded_file))
            st.success('Done!')
            st.balloons()
            st.download_button(label="Download Image", data='THE IMAGE RETURNED')
            
            
            


if __name__ == '__main__':
    main()