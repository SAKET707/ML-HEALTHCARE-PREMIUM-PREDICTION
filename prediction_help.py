import pandas as pd
import joblib

model=joblib.load("ARTIFACTS/model.joblib")
scaler=joblib.load("ARTIFACTS/scaler.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure":6,
    "thyroid": 5,
    "no disease": 0,
    "none":0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    max_score = 14 
    min_score = 0 
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score
    
def handle_scaling(df):

    cols_to_scale1 = scaler['cols_to_scale']
    scaler1 = scaler['scaler']

    df['income_level'] = None 
    df[cols_to_scale1] = scaler.transform(df[cols_to_scale1])
    df.drop('income_level', axis='columns', inplace=True)

    return df

def calculate_ls_risk(physical_activity,stress_level):

    ls1 = physical_activity + stress_level
    life_style_risk_score = (ls1 - 0)/(8-0)
    return life_style_risk_score





def preprocess_input(input_dict):

    expected_columns = [
        'age', 'number_of_dependants','smoking_status' ,'income_lakhs', 'insurance_plan','life_style_risk_score','normalized_risk_score',
        'gender_male', 'region_northwest', 'region_southeast', 'region_southwest', 'marital_status_unmarried',
        'bmi_category_obesity', 'bmi_category_overweight', 'bmi_category_underweight',
        'employment_status_salaried', 'employment_status_self-employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    smoking_status_encoding = {'Regular':3,'Occasional':2,'No Smoking':1}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_northwest'] = 1
            elif value == 'Southeast':
                df['region_southeast'] = 1
            elif value == 'Southwest':
                df['region_southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_underweight'] = 1
        elif key == 'Smoking Status':
            df['smoking_status'] = smoking_status_encoding.get(value,1)
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_self-employed'] = 1
        elif key == 'Insurance Plan': 
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':  
            df['age'] = value
        elif key == 'Number of Dependants': 
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs': 
            df['income_lakhs'] = value

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df['life_style_risk_score'] = calculate_ls_risk(input_dict['Physical Activity'],input_dict['Stress Level'])
    df = handle_scaling(df)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)
    prediction = model.predict(input_df)
    return int(prediction[0])

