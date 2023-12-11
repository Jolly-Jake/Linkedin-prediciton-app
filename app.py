import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('trained_model.pkl')

st.title('Magic 8 ball guesses if you use linkedin!')
st.write('Input your information below.')

income_ranges_display = {
    'Less than $10,000': 1,
    '10 to under $20,000': 2,
    '20 to under $30,000': 3,
    '30 to under $40,000': 4,
    '40 to under $50,000': 5,
    '50 to under $75,000': 6,
    '75 to under $100,000': 7,
    '100 to under $150,000': 8,
    '$150,000 or more': 9
}

education_levels_display = {
    'Less than high school': 1,
    'High school incomplete': 2,
    'High school graduate': 3,
    'Some college, no degree': 4,
    'Two-year associate degree': 5,
    'Four-year college or university degree': 6,
    'Some postgraduate or professional schooling': 7,
    'Postgraduate or professional degree': 8
}



age = st.slider('Age', min_value=18, max_value=98, value=30)
selected_income_display = st.selectbox('Income', list(income_ranges_display.keys()), 3)  

income_numeric = income_ranges_display[selected_income_display]

selected_education_display = st.selectbox('Education Level', list(education_levels_display.keys()), 3)
education_numeric = education_levels_display[selected_education_display]

marital_status = st.selectbox('Marital Status', ('Married', 'Single'))
gender = st.radio('Gender', ('Male', 'Female'))


marital_mapping = {'Married': 1, 'Single': 0}
gender_mapping = {'Male': 1, 'Female': 2}
marital_code = marital_mapping[marital_status]
gender_code = gender_mapping[gender]

user_input = np.array([[income_numeric, education_numeric, 0, marital_code, gender_code, age]])

user_prediction = model.predict_proba(user_input)[:, 1]

if st.button('Find out!'):
    
    user_input = np.array([[income_numeric, education_numeric, 0, marital_code, gender_code, age]])

    predicted_probability = model.predict_proba(user_input)[:, 1]
    threshold = 0.5
    if predicted_probability >= threshold:
        output_label = 1
    else:
        output_label = 2

    st.write('Prediction:')
    st.write('LinkedIn User: ' + str(output_label))
    st.write('Probability: ' + str(predicted_probability))
