import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import time
from PIL import Image
import os
from io import BytesIO

header = st.container()

dataset = st.container()

features = st.container()

modelTraining = st.container()

model = st.container()

st.sidebar.title("Sat Team")
st.sidebar.write("Rajat Sehgal")
st.sidebar.write("Francisco Varela Cid")
st.sidebar.write("Tofunmi Oludare")
st.sidebar.write("Shamil Aliyev")

with header:
    st.title("SATELLITE IMAGE CLASSIFIER by Sat Team")
    st.text("Here you can upload a satellite image and identify the objects in it.")



