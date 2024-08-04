

import numpy as np
import pickle
import pandas as pd

import streamlit as st 

# from PIL import Image

# #app=Flask(__name__)
# #Swagger(app)

pickle_in = open("model_pickle_new","rb")
model=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_ML_score(Libraries,Statistics,Basic_maths,supervised_algorithms,unsupervised_algorithms,semi_supervised_algorithms,reinforced_algorithm):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=model.predict([[Libraries,Statistics,Basic_maths,supervised_algorithms,unsupervised_algorithms,semi_supervised_algorithms,reinforced_algorithm]])
    print(prediction)
    return prediction



def main():
    st.title("ML SCORE PREDICTOR")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Libraries = st.text_input("Libraries","Type Here")
    Statistics = st.text_input("Statistics","Type Here")
    Basic_maths = st.text_input("Basic_maths","Type Here")
    supervised_algorithms = st.text_input("supervised_algorithms","Type Here")
    unsupervised_algorithms = st.text_input("unsupervised_algorithms","Type Here")
    semi_supervised_algorithms = st.text_input("semi_supervised_algorithms","Type Here")
    reinforced_algorithm = st.text_input("reinforced_algorithm","Type Here")

    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Libraries,Statistics,Basic_maths,supervised_algorithms,unsupervised_algorithms,semi_supervised_algorithms,reinforced_algorithm)
    st.success('The output is {}'.format(result))
    

if __name__=='__main__':
    main()
    
    
 