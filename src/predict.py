import joblib
import pandas as pd
import numpy as np
import os
def predict_single_customer(customer_dict):
    # Load comparison results to find the best model
    df_results = pd.read_csv('results/comparison_results.csv')
    best_model_name = df_results.iloc[0]['Model']
    print(f"Loading Best Model: {best_model_name}")
    
    if best_model_name == 'ANN':
        import tensorflow as tf
        model = tf.keras.models.load_model('saved_models/ANN.keras')
    else:
        model = joblib.load(f'saved_models/{best_model_name}.pkl')
        
    scaler = joblib.load('saved_models/scaler.pkl')
    data = joblib.load('results/processed_data.pkl')
    feature_names = data[4]  # The 5th element in the tuple

    # Convert dict to DataFrame
    df = pd.DataFrame([customer_dict])
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float).fillna(0)
    
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # 70.35 is the median MonthlyCharges from the raw dataset
    df['high_monthly_charges'] = (df['MonthlyCharges'] > 70.35).astype(int) 
    df['is_senior_no_support'] = ((df['SeniorCitizen'] == 1) & (df['TechSupport'] == 'No')).astype(int)
    
    # Advanced Feature Engineering
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    df['TotalCharges'] = np.log1p(df['TotalCharges'])
    
    # Advanced Feature Engineering — Round 2 (business-logic signals)
    df['long_tenure_churner'] = ((df['tenure'] > 24) & (df['Contract'] == 'Month-to-month')).astype(int)
    df['high_pay_no_support'] = ((df['MonthlyCharges'] > 70.35) & (df['TechSupport'] == 'No')).astype(int)
    df['no_sticky_services'] = ((df['OnlineSecurity'] == 'No') & (df['OnlineBackup'] == 'No') & (df['DeviceProtection'] == 'No')).astype(int)
    df['risky_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    contract_months = df['Contract'].map(contract_map)
    df['tenure_contract_ratio'] = df['tenure'] / contract_months
    
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            if col == 'gender':
                df[col] = df[col].map({'Female': 0, 'Male': 1})
            else:
                df[col] = df[col].map({'Yes': 1, 'No': 0})

    multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=[c for c in multi_class_cols if c in df.columns], drop_first=False)
    
    # Align columns to what the model expects
    df = df.reindex(columns=feature_names, fill_value=0)
    
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure', 'total_services', 'tenure_contract_ratio']
    df[num_features] = scaler.transform(df[num_features])
    df.columns = df.columns.astype(str)

    if best_model_name == 'ANN':
        prob = float(model.predict(df, verbose=0).ravel()[0])
    else:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(df)[0, 1])
        else:
            prob_raw = float(model.decision_function(df)[0])
            import math
            prob = 1 / (1 + math.exp(-prob_raw))

    prediction = 1 if prob >= 0.5 else 0
    
    if prob < 0.3:
        risk = 'Low'
    elif prob <= 0.6:
        risk = 'Medium'
    else:
        risk = 'High'
        
    return {
        'churn_prediction': prediction,
        'churn_probability': prob,
        'risk_level': risk
    }

if __name__ == "__main__":
    test_customer = {
      'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes',
      'Dependents': 'No', 'tenure': 12, 'PhoneService': 'Yes',
      'MultipleLines': 'No', 'InternetService': 'Fiber optic',
      'OnlineSecurity': 'No', 'OnlineBackup': 'No',
      'DeviceProtection': 'No', 'TechSupport': 'No',
      'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
      'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
      'PaymentMethod': 'Electronic check',
      'MonthlyCharges': 85.5, 'TotalCharges': 1026.0
    }
    result = predict_single_customer(test_customer)
    print("\nPrediction Result:")
    print(result)
