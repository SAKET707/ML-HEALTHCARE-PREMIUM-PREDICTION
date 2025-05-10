import pandas as pd
import joblib


model = joblib.load("ARTIFACTS/model.joblib")
scaler = joblib.load("ARTIFACTS/scaler.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)
    return total_risk_score / 14  

def calculate_ls_risk(physical_activity, stress_level):
    phy = {"Low": 4, "Medium": 1, "High": 0}
    stress = {"Low": 0, "Medium": 1, "High": 4}
    score = phy.get(physical_activity, 0) + stress.get(stress_level, 0)
    return score / 8  

def handle_scaling(df):
    cols_to_scale = scaler['cols_to_scale']
    scaler_model = scaler['scaler']

    if 'income_level' not in df.columns:
        print("Adding dummy 'income_level' column.")
        df['income_level'] = 0  

    df[cols_to_scale] = scaler_model.transform(df[cols_to_scale])

    return df



def preprocess_input(input_dict):
   
    feature_columns = [
        'age', 'number_of_dependants', 'smoking_status', 'income_lakhs', 'insurance_plan',
        'gender_male', 'region_northwest', 'region_southeast', 'region_southwest',
        'marital_status_unmarried',
        'bmi_category_obesity', 'bmi_category_overweight', 'bmi_category_underweight',
        'employment_status_salaried', 'employment_status_self-employed',
        'life_style_risk_score', 'normalized_risk_score'
    ]
    df = pd.DataFrame(0, index=[0], columns=feature_columns)

  
    insurance_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    smoking_map = {'No Smoking': 1, 'Occasional': 2, 'Regular': 3}

 
    df.at[0, 'age'] = input_dict['Age']
    df.at[0, 'number_of_dependants'] = input_dict['Number of Dependants']
    df.at[0, 'income_lakhs'] = input_dict['Income in Lakhs']
    df.at[0, 'smoking_status'] = smoking_map.get(input_dict['Smoking Status'], 1)
    df.at[0, 'insurance_plan'] = insurance_map.get(input_dict['Insurance Plan'], 1)

 
    if input_dict['Gender'] == 'Male':
        df.at[0, 'gender_male'] = 1
    if input_dict['Region'] == 'Northwest':
        df.at[0, 'region_northwest'] = 1
    elif input_dict['Region'] == 'Southeast':
        df.at[0, 'region_southeast'] = 1
    elif input_dict['Region'] == 'Southwest':
        df.at[0, 'region_southwest'] = 1
    if input_dict['Marital Status'] == 'Unmarried':
        df.at[0, 'marital_status_unmarried'] = 1
    if input_dict['BMI Category'] == 'Obesity':
        df.at[0, 'bmi_category_obesity'] = 1
    elif input_dict['BMI Category'] == 'Overweight':
        df.at[0, 'bmi_category_overweight'] = 1
    elif input_dict['BMI Category'] == 'Underweight':
        df.at[0, 'bmi_category_underweight'] = 1
    if input_dict['Employment Status'] == 'Salaried':
        df.at[0, 'employment_status_salaried'] = 1
    elif input_dict['Employment Status'] == 'Self-Employed':
        df.at[0, 'employment_status_self-employed'] = 1


    df.at[0, 'life_style_risk_score'] = calculate_ls_risk(input_dict['Physical Activity'], input_dict['Stress Level'])
    df.at[0, 'normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])


    df = handle_scaling(df)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)


    expected_cols = model.get_booster().feature_names
    input_df = input_df[expected_cols]

    prediction = model.predict(input_df)
    return int(prediction[0])
