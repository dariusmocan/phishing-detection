import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

def main():
    # Setup paths (train.py is in src/models/, so we go up 3 levels to the root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_CSV = os.path.join(BASE_DIR, "data", "features", "features_matrix.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please run features.py first.")
        return
        
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Split into features (X) and target (y)
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Phase 5: Train-Test Split (80% Train, 20% Test)
    # Using stratify to maintain the phishing/legitimate ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scale features using MinMaxScaler instead of StandardScaler!
    # MinMaxScaler scales everything between 0 and 1 (no negative numbers).
    # This stops `body_length` doing thousands of "counts" and crushing TF-IDF scores,
    # while allowing MultinomialNB to actually accept the scaled data!
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the test set arrays and the scaler for the evaluation & prediction phases
    joblib.dump((X_test, y_test), os.path.join(MODEL_DIR, "test_data.pkl"))
    joblib.dump((X_test_scaled, y_test), os.path.join(MODEL_DIR, "test_data_scaled.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\nTraining Models...")
    
    # ---------------------------------------------------------
    # 1. Naive Bayes 
    # ---------------------------------------------------------
    print(" -> Training Naive Bayes (MultinomialNB)...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_scaled, y_train)  # Now properly using scaled data!
    joblib.dump(nb_model, os.path.join(MODEL_DIR, "model_nb.pkl"))
    
    # ---------------------------------------------------------
    # 2. Logistic Regression
    # ---------------------------------------------------------
    print(" -> Training Logistic Regression (with scaled features)...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "model_logreg.pkl"))
    
    # ---------------------------------------------------------
    # 3. Random Forest
    # ---------------------------------------------------------
    print(" -> Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)  # Training on scaled for consistency with LR test set later
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "model_rf.pkl"))
    
    print("\n==================================================")
    print("Phase 5 Complete: All Models Trained & Serialized!")
    print(f"Check the `{os.path.relpath(MODEL_DIR)}` directory for the `.pkl` artifacts.")
    print("==================================================")

if __name__ == "__main__":
    main()