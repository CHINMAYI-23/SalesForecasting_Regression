import streamlit as st
import numpy as np
import pandas as pd
from sklearn import preprocessing
from PIL import Image
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.title('Sales Forecasting')
st.sidebar.header('Data')

# Function
def user_report():
    TV = st.sidebar.slider('TV Price',1,300,1)
    Radio = st.sidebar.slider('Radio Price',1,300,1)
    Newspaper = st.sidebar.slider('Newspaper Price',1,300,1)

    user_report_data = {
        'TV':TV,
        'Radio':Radio,
        'Newspaper':Newspaper
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()
st.header('Sales Data')
st.write(user_data)

Sales = model.predict(user_data)
st.subheader('Sales Prices')
st.subheader(np.round(Sales[0], 2))
