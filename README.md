# ML-HEALTHCARE-PREMIUM-PREDICTION
ML-HEALTHCARE-PREMIUM-PREDICTION

************************************************************************************************************************************************************************************************
URL : https://ml-healthcare-premium-prediction-by-saket.streamlit.app/
************************************************************************************************************************************************************************************************

This project uses machine learning to predict the healthcare insurance premium based on various factors such as medical history, lifestyle, age, BMI, employment status, and more.

## Project Overview

This application is built using a machine learning model, `XGBoost`, and is deployed with the help of Streamlit for easy user interaction. The user can input details related to their health, demographics, and lifestyle, and the model predicts the healthcare premium cost.

## Features

* **Predicts Healthcare Premium:** Given inputs related to age, income, medical history, BMI, and lifestyle, the model predicts the healthcare insurance premium cost.
* **Medical History:** The model considers diseases such as diabetes, heart disease, high blood pressure, and thyroid conditions.
* **Lifestyle Risk Score:** Based on physical activity and stress levels, the model calculates a risk score to determine the impact on the premium.
* **Dynamic Input Processing:** Users can provide input through an interactive form in the Streamlit app.


## Files and Structure

* **main.py:** The entry point of the Streamlit app where the user inputs data and sees predictions.
* **prediction\_help.py:** Contains functions for preprocessing input data, scaling features, and predicting the output.
* **model.joblib:** The trained machine learning model (`XGBoost`) used for prediction.
* **scaler.joblib:** The scaler used for scaling input features before prediction.
* **requirements.txt:** Contains a list of required Python libraries for the project.

## Input Fields

The following input fields are required to predict the healthcare premium:

* **Age:** The age of the user.
* **Number of Dependants:** Number of dependants the user is financially responsible for.
* **Smoking Status:** Whether the user smokes regularly, occasionally, or not at all.
* **Income in Lakhs:** Annual income in lakhs.
* **Insurance Plan:** The type of insurance plan (e.g., Bronze, Silver, Gold).
* **Medical History:** A string representing any diseases the user has (e.g., "diabetes & heart disease").
* **Physical Activity:** The user's level of physical activity (Low, Medium, High).
* **Stress Level:** The user's stress level (Low, Medium, High).
* **BMI Category:** The user's Body Mass Index (BMI) category (Obesity, Overweight, Underweight, Normal).
* **Marital Status:** Whether the user is married or unmarried.
* **Employment Status:** The user's employment status (Salaried, Self-Employed, Freelancer).
* **Region:** The region where the user resides (Northeast ,Northwest, Southeast, Southwest).

## Model and Approach

The model used for prediction is based on **XGBoost**, a gradient boosting model. Here's how it works:

1. **Data Preprocessing:** The input data undergoes various preprocessing steps, including encoding categorical variables, calculating lifestyle and health risk scores, and scaling numerical features.
2. **Feature Engineering:** Medical history is used to calculate a normalized risk score, and lifestyle factors like physical activity and stress are used to generate a risk score.
3. **Prediction:** The preprocessed data is passed through the trained model to predict the healthcare premium cost.

### Normalized Risk Score

The **normalized risk score** is calculated based on the user's medical history. Each disease is assigned a risk score, and the final score is normalized between 0 and 1.

### Lifestyle Risk Score

The **lifestyle risk score** is based on the user's physical activity and stress level. Each factor contributes to a total risk score, which is then normalized between 0 and 1.


************************************************************************************************************************************************************************************************

