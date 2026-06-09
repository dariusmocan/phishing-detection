import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def print_metrics(model_name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("-" * 50)
    print(f" Model: {model_name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC:   {auc:.4f}")
        
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
    
    unscaled_data_path = os.path.join(MODEL_DIR, "test_data.pkl")
    scaled_data_path = os.path.join(MODEL_DIR, "test_data_scaled.pkl")
    
    if not os.path.exists(unscaled_data_path) or not os.path.exists(scaled_data_path):
        print("Error: Test data missing. Please run src/models/train.py first.")
        return
        
    print("Loading test datasets and models...")
    X_test_unscaled, y_test = joblib.load(unscaled_data_path)
    X_test_scaled, _ = joblib.load(scaled_data_path)
    
    nb_model = joblib.load(os.path.join(MODEL_DIR, "model_nb.pkl"))
    lr_model = joblib.load(os.path.join(MODEL_DIR, "model_logreg.pkl"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "model_rf.pkl"))
    
    print("\n" + "="*50)
    print(" PHASE 6: MACHINE LEARNING EVALUATION")
    print("="*50)
    
    # 1. Naive Bayes (Now using correctly scaled MinMax data!)
    nb_pred = nb_model.predict(X_test_scaled)
    nb_prob = nb_model.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Naive Bayes", y_test, nb_pred, nb_prob)
    
    # 2. Logistic Regression (Uses scaled data)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Logistic Regression", y_test, lr_pred, lr_prob)
    
    # 3. Random Forest (Trained on scaled data for this pipeline)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Random Forest", y_test, rf_pred, rf_prob)
    
    print("\n==================================================")
    print("Evaluation Complete. Compare these to the Rule-Based Baseline!")
    print("==================================================")

if __name__ == "__main__":
    main()