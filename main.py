import streamlit as st
from prediction_help import predict

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.title("HEALTHCARE PREMIUM PREDICTION")

categorical_options = {
    'Gender': ['Female', 'Male'],
    'Region': ['Southeast', 'Northeast', 'Southwest', 'Northwest'],
    'Marital_status': ['Unmarried', 'Married'],
    'BMI_Category': ['Normal', 'Overweight', 'Obesity', 'Underweight'],
    'Smoking_Status': ['Regular', 'Occasional', 'No Smoking'],
    'Employment_Status': ['Self-Employed', 'Freelancer', 'Salaried'],
    'Medical_History': [
        'High blood pressure', 'No Disease', 'Thyroid',
        'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes', 'Heart disease', 'Diabetes & High blood pressure',
        'Diabetes & Heart disease'
    ],
    "Insurance_Plan": ['Gold', 'Silver', 'Bronze'],
    'Physical_Activity': ['Low', 'Medium', 'High'],
    'Stress_Level': ['Low', 'Medium', 'High']
}

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(1)

with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100)
with row1[1]:
    gender = st.selectbox('Gender', categorical_options['Gender'])
with row1[2]:
    region = st.selectbox('Region', categorical_options['Region'])

with row2[0]:
    physical_activity = st.selectbox('Physical Activity', categorical_options['Physical_Activity'])
with row2[1]:
    stress_level = st.selectbox('Stress Level', categorical_options['Stress_Level'])
with row2[2]:
    number_of_dependants = st.number_input('Number of Dependants', min_value=0, step=1, max_value=10)

with row3[0]:
    bmi_category = st.selectbox('BMI Category', categorical_options['BMI_Category'])
with row3[1]:
    smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking_Status'])
with row3[2]:
    employment_status = st.selectbox('Employment Status', categorical_options['Employment_Status'])

with row4[0]:
    income_lakhs = st.number_input('Income in Lakhs', step=1, min_value=0, max_value=200)
with row4[1]:
    medical_history = st.selectbox('Medical History', categorical_options['Medical_History'])
with row4[2]:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance_Plan'])

with row5[0]:
    marital_status = st.selectbox("Marital Status", categorical_options['Marital_status'])

input_dict = {
    'Age': age,
    'Gender': gender,
    'Region': region,
    'Physical Activity': physical_activity,
    'Stress Level': stress_level,
    'Number of Dependants': number_of_dependants,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Employment Status': employment_status,
    'Income in Lakhs': income_lakhs,
    'Medical History': medical_history,
    'Insurance Plan': insurance_plan,
    'Marital Status': marital_status
}

if st.button('Predict'):
    prediction = predict(input_dict)
    st.success(f'Predicted Health Insurance Cost: {prediction}')
