import joblib
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_all_models():
    os.makedirs('saved_models', exist_ok=True)
    
    print("Loading processed data...")
    data = joblib.load('results/processed_data.pkl')
    # Unpack the tuple
    X_train, X_test, y_train, y_test, feature_names = data
    
    # ═══════════════════════════════════
    # MODEL 1: LOGISTIC REGRESSION (baseline)
    # ═══════════════════════════════════
    print("Training Logistic Regression...")
    start_time = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'saved_models/Logistic_Regression.pkl')
    print(f"Logistic Regression training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # MODEL 2: DECISION TREE
    # ═══════════════════════════════════
    print("Training Decision Tree...")
    start_time = time.time()
    dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    joblib.dump(dt, 'saved_models/Decision_Tree.pkl')
    print(f"Decision Tree training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # MODEL 3: RANDOM FOREST
    # ═══════════════════════════════════
    print("Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42,
                           class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'saved_models/Random_Forest.pkl')
    print(f"Random Forest training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # MODEL 4: SUPPORT VECTOR MACHINE
    # ═══════════════════════════════════
    print("Training Support Vector Machine...")
    start_time = time.time()
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42,
        class_weight='balanced')
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'saved_models/SVM.pkl')
    print(f"SVM training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # MODEL 5: XGBOOST
    # ═══════════════════════════════════
    print("Training XGBoost...")
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric='logloss',
                      random_state=42)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'saved_models/XGBoost.pkl')
    print(f"XGBoost training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # MODEL 6: LIGHTGBM
    # ═══════════════════════════════════
    print("Training LightGBM...")
    start_time = time.time()
    lgbm_model = lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                       num_leaves=31, random_state=42, class_weight='balanced')
    lgbm_model.fit(X_train, y_train)
    joblib.dump(lgbm_model, 'saved_models/LightGBM.pkl')
    print(f"LightGBM training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # MODEL 7: ARTIFICIAL NEURAL NETWORK (ANN)
    # ═══════════════════════════════════
    print("Training Artificial Neural Network (ANN)...")
    start_time = time.time()
    ann_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann_model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    ann_model.fit(X_train, y_train, epochs=100, batch_size=64,
              validation_split=0.2, callbacks=callbacks, verbose=1,
              class_weight={0: 1.0, 1: 2.7})  # reflect 73/27 class imbalance
    ann_model.save('saved_models/ANN.keras')
    print(f"ANN training time: {time.time() - start_time:.2f} seconds\n")

    # ═══════════════════════════════════
    # MODEL 8: CATBOOST
    # ═══════════════════════════════════
    print("Training CatBoost...")
    start_time = time.time()
    cat_model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights='Balanced', random_seed=42, verbose=0
    )
    cat_model.fit(X_train, y_train)
    joblib.dump(cat_model, 'saved_models/CatBoost.pkl')
    print(f"CatBoost training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # MODEL 8: DIVERSE STACKING CLASSIFIER
    # ═══════════════════════════════════
    print("Training Diverse Stacking Classifier...")
    start_time = time.time()
    # 5 diverse base models (3 tree + 1 linear + 1 kernel) for better generalisation
    base_models = [
        ('rf',  RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, num_leaves=31, random_state=42, class_weight='balanced')),
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', SVC(probability=True, kernel='rbf', random_state=42))
    ]
    # passthrough=True feeds original features to meta-model alongside base predictions
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True
    )
    stacking_model.fit(X_train, y_train)
    joblib.dump(stacking_model, 'saved_models/Stacking_Ensemble.pkl')
    print(f"Stacking Classifier training time: {time.time() - start_time:.2f} seconds\n")

    # ═══════════════════════════════════
    # MODEL 9: CALIBRATED STACKING
    # ═══════════════════════════════════
    print("Training Calibrated Stacking Classifier...")
    start_time = time.time()
    calibrated_stack = CalibratedClassifierCV(
        StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, passthrough=True
        ),
        method='isotonic', cv=3
    )
    calibrated_stack.fit(X_train, y_train)
    joblib.dump(calibrated_stack, 'saved_models/Stacking_Calibrated.pkl')
    print(f"Calibrated Stacking training time: {time.time() - start_time:.2f} seconds\n")
    
    # ═══════════════════════════════════
    # BAYESIAN OPTIMIZATION WITH OPTUNA
    # ═══════════════════════════════════
    print("Starting Bayesian Optimization with Optuna...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # XGBoost Optimization
    print("Optimizing XGBoost...")
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        return scores.mean()

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=50)
    
    print(f"Best XGBoost Params: {study_xgb.best_params}")
    print(f"Best XGBoost F1 Score: {study_xgb.best_value:.4f}")
    
    best_xgb = xgb.XGBClassifier(**study_xgb.best_params, eval_metric='logloss', random_state=42)
    best_xgb.fit(X_train, y_train)
    joblib.dump(best_xgb, 'saved_models/XGBoost_Optimized.pkl')
    
    # LightGBM Optimization
    print("Optimizing LightGBM...")
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'class_weight': 'balanced',
            'random_state': 42
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        return scores.mean()

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=50)
    
    print(f"Best LightGBM Params: {study_lgb.best_params}")
    print(f"Best LightGBM F1 Score: {study_lgb.best_value:.4f}")
    
    best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, class_weight='balanced', random_state=42)
    best_lgb.fit(X_train, y_train)
    joblib.dump(best_lgb, 'saved_models/LightGBM_Optimized.pkl')

    # Full-Stack Optuna: tune RF + XGBoost params inside stacking
    print("Optimizing Full Stacking Pipeline with Optuna...")
    def objective_stack(trial):
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 400),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 12),
        }
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('xgb_depth', 3, 8),
            'eval_metric': 'logloss'
        }
        stack = StackingClassifier(
            estimators=[
                ('rf',  RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)),
                ('xgb', xgb.XGBClassifier(**xgb_params, random_state=42)),
                ('lgb', lgb.LGBMClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
                ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, passthrough=True
        )
        scores = cross_val_score(stack, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        return scores.mean()

    study_stack = optuna.create_study(direction='maximize')
    study_stack.optimize(objective_stack, n_trials=30)
    print(f"Best Stack Params: {study_stack.best_params}")
    print(f"Best Stack F1: {study_stack.best_value:.4f}")

    p = study_stack.best_params
    best_stack = StackingClassifier(
        estimators=[
            ('rf',  RandomForestClassifier(n_estimators=p['rf_n_estimators'], max_depth=p['rf_max_depth'], random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=p['xgb_n_estimators'], learning_rate=p['xgb_lr'], max_depth=p['xgb_depth'], eval_metric='logloss', random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
            ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=True
    )
    best_stack.fit(X_train, y_train)
    joblib.dump(best_stack, 'saved_models/Stacking_Optimized.pkl')

    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    train_all_models()
