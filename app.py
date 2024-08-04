import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model_pickle_new', 'rb') as file:
    model = pickle.load(file)

st.title('Machine Learning Score Predictor')

st.sidebar.header('Input Features')
libraries = st.sidebar.selectbox('Libraries', [0, 1])
statistics = st.sidebar.selectbox('Statistics', [0, 1])
basic_maths = st.sidebar.selectbox('Basic Maths', [0, 1])
supervised_algorithms = st.sidebar.selectbox('Supervised Algorithms', [0, 1])
unsupervised_algorithms = st.sidebar.selectbox('Unsupervised Algorithms', [0, 1])
semi_supervised_algorithms = st.sidebar.selectbox('Semi-Supervised Algorithms', [0, 1])
reinforced_algorithm = st.sidebar.selectbox('Reinforced Algorithm', [0, 1])

input_data = pd.DataFrame({
    'Libraries': [libraries],
    'Statistics': [statistics],
    'Basic maths': [basic_maths],
    'supervised algorithms': [supervised_algorithms],
    'unsupervised algorithms': [unsupervised_algorithms],
    'semi-supervised algorithms': [semi_supervised_algorithms],
    'reinforced algorithm': [reinforced_algorithm]
})

prediction = model.predict(input_data)

st.subheader('Predicted ML Score')
st.write(prediction[0])
