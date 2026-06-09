import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def heuristic_rules(row):
    """
    Simulates a non-ML, deterministic threat detection engine.
    If any of these obvious red flags trigger, it flags as Phishing (1), else Legitimate (0).
    """
    # Rule 1: IP in URL is a classic indicator of evasion
    if row.get('url_has_ip', 0) == 1:
        return 1
        
    # Rule 2: Usage of URL shorteners to hide final destination
    if row.get('url_has_shortener', 0) == 1:
        return 1
        
    # Rule 3: High urgency subject lines (often used in social engineering)
    if row.get('subj_urgency', 0) == 1:
        return 1
        
    # Default to Legitimate (0) if no rules triggered
    return 0

def main():
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_CSV = os.path.join(BASE_DIR, "data", "features", "features_matrix.csv")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Ensure features.py has been run.")
        return
        
    print(f"Loading {INPUT_CSV} to evaluate non-ML baselines...")
    df = pd.read_csv(INPUT_CSV)
    
    # Ground truth labels
    y_true = df['label']
    
    # Apply baseline rules row-by-row
    print("Applying deterministic heuristics (Sender, URL, and Content strings)...")
    y_pred = df.apply(heuristic_rules, axis=1)
    
    # Calculate performance floor metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Display results
    print("\n" + "="*50)
    print("NON-ML BASELINE (RULE-BASED) PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}  (Overall correctness)")
    print(f"Precision: {prec:.4f}  (When it flagged Phishing, was it right?)")
    print(f"Recall:    {rec:.4f}  (How much total Phishing did it catch?)")
    print(f"F1-Score:  {f1:.4f}  (Harmonic mean of Precision/Recall)")
    print("-" * 50)
    print("Confusion Matrix (TN  FP | FN  TP):")
    print(conf_matrix)
    print("==================================================")
    print("This defines the performance floor. Our ML models must beat these numbers!")

if __name__ == "__main__":
    main()