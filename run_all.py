import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess
from src.train import train_all_models
from src.evaluate import evaluate_all_models
from src.predict import predict_single_customer

if __name__ == '__main__':
    print('='*60)
    print('TELECOM CHURN PREDICTION - FULL PIPELINE')
    print('='*60)

    print('[1/4] Running preprocessing...')
    # Using data/raw/ since we moved the file there earlier
    preprocess(data_path='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    print('[2/4] Training all models...')
    train_all_models()

    print('[3/4] Evaluating models...')
    results = evaluate_all_models()
    print('Best model:', results.iloc[0]['Model'])

    print('[4/4] Testing single prediction...')
    test_customer = {
        'gender':'Male','SeniorCitizen':0,'Partner':'Yes',
        'Dependents':'No','tenure':12,'PhoneService':'Yes',
        'MultipleLines':'No','InternetService':'Fiber optic',
        'OnlineSecurity':'No','OnlineBackup':'No',
        'DeviceProtection':'No','TechSupport':'No',
        'StreamingTV':'Yes','StreamingMovies':'Yes',
        'Contract':'Month-to-month','PaperlessBilling':'Yes',
        'PaymentMethod':'Electronic check',
        'MonthlyCharges':85.5,'TotalCharges':1026.0
    }
    result = predict_single_customer(test_customer)
    print('Prediction result:', result)
    print('='*60)
    print('PIPELINE COMPLETE. Check results/ folder for all outputs.')
    print('='*60)
