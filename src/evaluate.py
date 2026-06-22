import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss)
import tensorflow as tf

def evaluate_all_models():
    print("Loading test data...")
    data = joblib.load('results/processed_data.pkl')
    X_train, X_test, y_train, y_test, feature_names = data

    model_names = [
        'Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'SVM',
        'XGBoost', 'LightGBM', 'XGBoost_Optimized', 'LightGBM_Optimized',
        'CatBoost', 'Stacking_Ensemble', 'Stacking_Calibrated', 'Stacking_Optimized', 'ANN'
    ]

    models = {}
    print("Loading models...")
    for name in model_names:
        if name == 'ANN':
            models[name] = tf.keras.models.load_model('saved_models/ANN.keras')
        else:
            models[name] = joblib.load(f'saved_models/{name}.pkl')

    results = []
    roc_data = {}
    pr_data = {}
    cm_data = {}
    
    print("Evaluating models...")
    for name, model in models.items():
        if name == 'ANN':
            y_pred_prob = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_pred_prob >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_prob = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_prob)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC-ROC': auc
        })

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_data[name] = (fpr, tpr, auc)
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_data[name] = (precision, recall)

        cm_data[name] = confusion_matrix(y_test, y_pred)

    df_results = pd.DataFrame(results).sort_values(by='F1', ascending=False)

    # Threshold tuning on best sklearn model (Stacking_Ensemble)
    print("\nRunning threshold tuning on Stacking_Ensemble...")
    stack_probs = models['Stacking_Ensemble'].predict_proba(X_test)[:, 1]
    precisions_t, recalls_t, thresholds_t = precision_recall_curve(y_test, stack_probs)
    f1_scores_t = 2 * (precisions_t * recalls_t) / (precisions_t + recalls_t + 1e-8)
    best_thresh = thresholds_t[np.argmax(f1_scores_t[:-1])]
    y_pred_tuned = (stack_probs >= best_thresh).astype(int)
    tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted')
    print(f"Optimal threshold: {best_thresh:.3f} | Default F1: {df_results[df_results.Model=='Stacking_Ensemble']['F1'].values[0]:.4f} | Tuned F1: {tuned_f1:.4f}")

    # Brier score for calibrated model
    cal_probs = models['Stacking_Calibrated'].predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, cal_probs)
    print(f"Stacking_Calibrated Brier Score: {brier:.4f} (lower = better calibrated)")

    # Save CSV
    df_results.to_csv('results/comparison_results.csv', index=False)
    print("Saved comparison_results.csv")

    # 1. Model Comparison Bar Chart
    df_melt = df_results.melt(id_vars='Model', value_vars=['Accuracy', 'F1', 'AUC-ROC'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric')
    plt.xticks(rotation=45)
    plt.title('Model Comparison')
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/roc_curves.png')
    plt.close()

    # 3. Confusion Matrices
    rows = int(np.ceil(len(model_names) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        sns.heatmap(cm_data[name], annot=True, fmt='d', ax=axes[idx], cmap='Blues')
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    for idx in range(len(model_names), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.close()

    # 4. Feature Importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    rf_model = models['Random_Forest']
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)[:15]
    sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=ax1, color='steelblue')
    ax1.set_title('Random Forest Feature Importance')

    xgb_opt_model = models['XGBoost_Optimized']
    xgb_imp = pd.Series(xgb_opt_model.feature_importances_, index=feature_names).sort_values(ascending=False)[:15]
    sns.barplot(x=xgb_imp.values, y=xgb_imp.index, ax=ax2, color='teal')
    ax2.set_title('XGBoost Optimized Feature Importance')

    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()

    # 5. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for name, (precision, recall) in pr_data.items():
        plt.plot(recall, precision, label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/precision_recall_curves.png')
    plt.close()

    print("Evaluation completed. Visualizations saved in 'results/'.")
    return df_results

if __name__ == "__main__":
    evaluate_all_models()
