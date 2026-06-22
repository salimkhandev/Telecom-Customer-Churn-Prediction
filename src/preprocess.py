import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess(data_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. LOAD DATA
    print("1. LOADING DATA...")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    print("Data types:\n", df.dtypes)
    print("Churn value counts:\n", df['Churn'].value_counts())
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # 2. HANDLE MISSING VALUES
    print("\n2. HANDLING MISSING VALUES...")
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    median_charges = df['TotalCharges'].median()
    
    print("Nulls before filling TotalCharges:", df['TotalCharges'].isnull().sum())
    df['TotalCharges'] = df['TotalCharges'].fillna(median_charges)
    print("Nulls after filling TotalCharges:", df['TotalCharges'].isnull().sum())
    
    # 3. ENCODE TARGET
    print("\n3. ENCODING TARGET...")
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    # 4. FEATURE ENGINEERING
    print("\n4. FEATURE ENGINEERING...")
    X['charges_per_tenure'] = X['MonthlyCharges'] / (X['tenure'] + 1)
    X['high_monthly_charges'] = (X['MonthlyCharges'] > X['MonthlyCharges'].median()).astype(int)
    X['is_senior_no_support'] = ((X['SeniorCitizen'] == 1) & (X['TechSupport'] == 'No')).astype(int)
    
    # Advanced Feature Engineering — Round 1
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    X['total_services'] = X[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    X['TotalCharges'] = np.log1p(X['TotalCharges'])
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    contract_months = X['Contract'].map(contract_map)
    X['tenure_contract_ratio'] = X['tenure'] / contract_months

    # Advanced Feature Engineering — Round 2 (business-logic signals)
    X['long_tenure_churner'] = ((X['tenure'] > 24) & (X['Contract'] == 'Month-to-month')).astype(int)
    X['high_pay_no_support'] = ((X['MonthlyCharges'] > X['MonthlyCharges'].median()) & (X['TechSupport'] == 'No')).astype(int)
    X['no_sticky_services'] = ((X['OnlineSecurity'] == 'No') & (X['OnlineBackup'] == 'No') & (X['DeviceProtection'] == 'No')).astype(int)
    X['risky_payment'] = (X['PaymentMethod'] == 'Electronic check').astype(int)
    
    # 5. ENCODE CATEGORICAL FEATURES
    print("\n5. ENCODING CATEGORICAL FEATURES...")
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col == 'gender':
            X[col] = X[col].map({'Female': 0, 'Male': 1})
        else:
            X[col] = X[col].map({'Yes': 1, 'No': 0})
            
    multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaymentMethod']
    X = pd.get_dummies(X, columns=multi_class_cols, drop_first=True)
    
    # 6. SCALE NUMERICAL FEATURES
    print("\n6. SCALING NUMERICAL FEATURES...")
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure', 'total_services', 'tenure_contract_ratio']
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    
    X.columns = X.columns.astype(str)

    # 8. TRAIN/TEST SPLIT
    print("\n8. TRAIN/TEST SPLIT...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. APPLY SMOTE FOR CLASS IMBALANCE
    print("\n7. APPLYING SMOTE...")
    print("Class distribution before SMOTE:\n", y_train.value_counts())
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:\n", y_train_resampled.value_counts())

    # 9. SAVE PROCESSED DATA
    print("\n9. SAVING PROCESSED DATA...")
    processed_data = (
        X_train_resampled,
        X_test,
        y_train_resampled,
        y_test,
        X_train_resampled.columns.tolist()
    )
    joblib.dump(processed_data, 'results/processed_data.pkl')
    
    # 10. VISUALIZATIONS
    print("\n10. GENERATING VISUALIZATIONS...")
    
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.tight_layout()
    plt.savefig('results/churn_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    X_corr = X.copy()
    X_corr['Churn'] = y
    corr = X_corr.corr()
    top_corr_features = corr.index[abs(corr['Churn']) > 0.1][:15] 
    sns.heatmap(X_corr[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title('Correlation Heatmap (Top 15)')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png')
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, bins=30)
    plt.title('Monthly Charges Distribution by Churn')
    plt.tight_layout()
    plt.savefig('results/monthly_charges_churn.png')
    plt.close()
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x='Churn', y='tenure')
    plt.title('Tenure vs Churn')
    plt.tight_layout()
    plt.savefig('results/tenure_churn_boxplot.png')
    plt.close()

    print("\nPreprocessing complete!")
    return X_train_resampled, X_test, y_train_resampled, y_test, X_train_resampled.columns.tolist()

if __name__ == "__main__":
    preprocess()
