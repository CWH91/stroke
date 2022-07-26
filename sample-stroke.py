import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

st.write("""
# Stroke Prediction App
This app predicts the possibility of stroke, depending on a multitude of health factors.""")

st.sidebar.header('Health Parameters')

def user_input_features():
    gender = st.sidebar.radio('Gender', ('Female','Male','Other'))
    age = st.sidebar.slider('Age', min_value=0, max_value=120, step=1)
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', min_value=50, max_value=300, step=0.1)
    hypertension = st.sidebar.radio('Hypertension', ('Yes','No',))
    heart_disease = st.sidebar.radio('Heart Disease', ('Yes','No',))
    smoking_status = st.sidebar.selectbox('Smoking Status', ('Unknown','formerly smoked','never smoked','smokes'))
    
    data = {'gender': gender,
            'age': age,
            'avg_glucose_level': avg_glucose_level,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_status': smoking_status,}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

stroke = pd.read_csv('https://github.com/CWH91/stroke/blob/main/healthcare-dataset-stroke-data.csv')
X = stroke.drop(['stroke','bmi'], axis = 1)

Y = stroke['stroke']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
