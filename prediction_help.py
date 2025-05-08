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
    normalized_risk_score = total_risk_score / 14 
    return normalized_risk_score

def calculate_ls_risk(physical_activity, stress_level):
    phy = {"Low": 4, "Medium": 1, "High": 0}
    stress = {"Low": 0, "Medium": 1, "High": 4}
    score = phy.get(physical_activity, 1) + stress.get(stress_level, 1)
    return score / 8

def handle_scaling(df):
    cols_to_scale = scaler['cols_to_scale']
    scale_model = scaler['scaler']
    df['income_level'] = None  
    df[cols_to_scale] = scale_model.transform(df[cols_to_scale])
    df.drop('income_level', axis=1, inplace=True)
    return df

def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'smoking_status', 'income_lakhs', 'insurance_plan',
        'life_style_risk_score', 'normalized_risk_score',
        'gender_male', 'region_northwest', 'region_southeast', 'region_southwest', 
        'marital_status_unmarried',
        'bmi_category_obesity', 'bmi_category_overweight', 'bmi_category_underweight',
        'employment_status_salaried', 'employment_status_self-employed'
    ]

    df = pd.DataFrame(0, index=[0], columns=expected_columns)

    df['age'] = input_dict.get('Age', 30)
    df['number_of_dependants'] = input_dict.get('Number of Dependants', 0)
    df['income_lakhs'] = input_dict.get('Income in Lakhs', 5)


    df['insurance_plan'] = {'Bronze': 1, 'Silver': 2, 'Gold': 3}.get(input_dict.get('Insurance Plan', 'Silver'), 2)
    df['smoking_status'] = {'Regular': 3, 'Occasional': 2, 'No Smoking': 1}.get(input_dict.get('Smoking Status', 'No Smoking'), 1)
    
    if input_dict.get('Gender') == 'Male':
        df['gender_male'] = 1
    if input_dict.get('Region') == 'Northwest':
        df['region_northwest'] = 1
    elif input_dict.get('Region') == 'Southeast':
        df['region_southeast'] = 1
    elif input_dict.get('Region') == 'Southwest':
        df['region_southwest'] = 1
    if input_dict.get('Marital Status') == 'Unmarried':
        df['marital_status_unmarried'] = 1

    bmi = input_dict.get('BMI Category')
    if bmi == 'Obesity':
        df['bmi_category_obesity'] = 1
    elif bmi == 'Overweight':
        df['bmi_category_overweight'] = 1
    elif bmi == 'Underweight':
        df['bmi_category_underweight'] = 1

    emp = input_dict.get('Employment Status')
    if emp == 'Salaried':
        df['employment_status_salaried'] = 1
    elif emp == 'Self-Employed':
        df['employment_status_self-employed'] = 1

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'No Disease'))
    df['life_style_risk_score'] = calculate_ls_risk(input_dict.get('Physical Activity', 'Medium'), input_dict.get('Stress Level', 'Medium'))

    df = handle_scaling(df)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)
    prediction = model.predict(input_df)
    return int(prediction[0])
