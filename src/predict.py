import pandas as pd
import argparse
import os
import sys
import joblib

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# We import the extraction functions from features.py
# Adding src to path if executed from root
sys.path.append(os.path.join(BASE_DIR, "src"))
try:
    from features import extract_header_features, extract_body_features, extract_url_features
    from rules import heuristic_rules
except ImportError:
    print("Failed to import feature extractors. Ensure 'src' is executable.")
    sys.exit(1)

def extract_live_features(email_dict):
    """
    Takes a mock raw email dictionary, passes it through the exact same
    3-layer extraction pipeline used in training, and returns the numeric vector.
    """
    # 1. Formatting exact schema that features.py expects
    df = pd.DataFrame([email_dict])
    
    # 2. Extract Layers
    df_headers = extract_header_features(df)
    df_urls = extract_url_features(df)
    
    # Body requires the pre-trained vectorizer
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Error: {VECTORIZER_PATH} missing.")
        sys.exit(1)
        
    df_body = extract_body_features(df, VECTORIZER_PATH)
    
    # 3. Concatenate (drop 'label' since it doesn't exist live)
    final_df = pd.concat([df_headers, df_body, df_urls], axis=1)
    
    # Ensure ordered exactly as the model expects
    return final_df

def predict_email(subject, body, urls):
    """
    Runs the inference pipeline across all four detection methods simultaneously.
    """
    email_data = {
        'subject': subject,
        'body': body,
        'urls': urls
    }
    
    # 1. Feature Extraction (Layer mapping)
    print("\nRunning extraction pipeline...")
    features = extract_live_features(email_data)
    
    # 2. Load Scaler & Data preparation
    scaler = joblib.load(SCALER_PATH)
    features_scaled = scaler.transform(features)
    
    # 3. Load Models
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'model_rf.pkl'))
    lr_model = joblib.load(os.path.join(MODEL_DIR, 'model_logreg.pkl'))
    nb_model = joblib.load(os.path.join(MODEL_DIR, 'model_nb.pkl'))
    
    # 4. Generate Predictions from all 4 systems
    
    # A) Heuristics
    heur_pred = heuristic_rules(features.iloc[0])
    
    # B) Naive Bayes
    nb_pred = nb_model.predict(features_scaled)[0]
    nb_prob = nb_model.predict_proba(features_scaled)[0]
    nb_conf = nb_prob[1] if nb_pred == 1 else nb_prob[0]
    
    # C) Logistic Regression (Requires scaled features)
    lr_pred = lr_model.predict(features_scaled)[0]
    lr_prob = lr_model.predict_proba(features_scaled)[0]
    lr_conf = lr_prob[1] if lr_pred == 1 else lr_prob[0]
    
    # D) Random Forest (Requires scaled features)
    rf_pred = rf_model.predict(features_scaled)[0]
    rf_prob = rf_model.predict_proba(features_scaled)[0]
    rf_conf = rf_prob[1] if rf_pred == 1 else rf_prob[0]
    
    # Helper formatter
    def format_pred(pred, conf=None):
        if pred == 1:
            return f"PHISHING" + (f" (Confidence: {conf*100:.1f}%)" if conf else "")
        return f"LEGITIMATE" + (f" (Confidence: {conf*100:.1f}%)" if conf else "")
    
    # 5. Output Grouped Explanations
    print("\n" + "="*60)
    print("MOCK EMAIL INFERENCE RESULTS - ALGORITHM COMPARISON")
    print("="*60)
    print(f"SUBJECT:   {subject}")
    display_body = body[:70] + "..." if len(body) > 70 else body
    print(f"BODY Snippet:{display_body}")
    print("-" * 60)
    
    print(f"1. Rule-Based Heuristics:")
    print(f" Prediction: {format_pred(heur_pred)}\n")
    
    print(f"2. Naive Bayes:")
    print(f" Prediction: {format_pred(nb_pred, nb_conf)}\n")
    
    print(f"3. Logistic Regression:")
    print(f" Prediction: {format_pred(lr_pred, lr_conf)}\n")
    
    print(f"4. Random Forest:")
    print(f" Prediction: {format_pred(rf_pred, rf_conf)}")
    print("=" * 60)
    

def main():
    parser = argparse.ArgumentParser(description="Phishing Email Classifier CLI")
    
    parser.add_argument("--subject", type=str, required=True, help="The subject line of the email")
    parser.add_argument("--body", type=str, required=True, help="The main body text of the email")
    parser.add_argument("--urls", type=str, default="", help="Extracted URLs (space separated)")

    args = parser.parse_args()
    
    predict_email(args.subject, args.body, args.urls)

if __name__ == "__main__":
    main()