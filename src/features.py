import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_header_features(df):
    print("Extracting Layer 1: Header Features...")
    # Fill NaN to avoid errors
    subjects = df['subject'].fillna("")
    
    df_feat = pd.DataFrame(index=df.index)
    
    subj_length_list = []
    subj_urgency_list = []
    subj_reply_list = []
    
    urgency_pattern = re.compile(r'(urgent|immediate|important|action required|suspended|verify|account|alert|security)', re.IGNORECASE)
    
    for subject in subjects:
        # 1. Subject Length
        subj_length_list.append(len(subject))
        
        # 2. Subject Urgency Words
        if urgency_pattern.search(subject):
            subj_urgency_list.append(1)
        else:
            subj_urgency_list.append(0)
            
        # 3. Reply/Forward (Legitimate emails often have these, but sometimes phishing spoofs them)
        if re.match(r'^(re|fwd|fw):', subject, re.IGNORECASE):
            subj_reply_list.append(1)
        else:
            subj_reply_list.append(0)
            
    df_feat['subj_length'] = subj_length_list
    df_feat['subj_urgency'] = subj_urgency_list
    df_feat['subj_is_reply_fwd'] = subj_reply_list
    
    return df_feat

def extract_url_features(df):
    print("Extracting Layer 3: URL Features...")
    urls_col = df['urls'].fillna("")
    
    df_feat = pd.DataFrame(index=df.index)
    
    url_count_list = []
    url_has_ip_list = []
    url_shortener_list = []
    
    shorteners = re.compile(r'(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|is\.gd|cli\.gs|tr\.im)', re.IGNORECASE)
    
    for url_text in urls_col:
        url_str = str(url_text)
        
        # 1. URL Count (Assuming URLs are present as a string of lists or separated strings)
        urls_found = re.findall(r'http[s]?://', url_str)
        url_count_list.append(len(urls_found))
        
        # 2. Presence of IP based URL (Standard phishing signal)
        if re.search(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_str):
            url_has_ip_list.append(1)
        else:
            url_has_ip_list.append(0)
        
        # 3. URL shorteners
        if shorteners.search(url_str):
            url_shortener_list.append(1)
        else:
            url_shortener_list.append(0)
            
    df_feat['url_count'] = url_count_list
    df_feat['url_has_ip'] = url_has_ip_list
    df_feat['url_has_shortener'] = url_shortener_list
    
    return df_feat

def extract_body_features(df, vectorizer_path):
    print("Extracting Layer 2: Body Features (NLP & Syntax)...")
    body = df['body'].fillna("")
    df_feat = pd.DataFrame(index=df.index)
    
    body_length_list = []
    body_has_html_list = []
    
    for b in body:
        body_str = str(b)
        
        # 1. Body Length
        body_length_list.append(len(body_str))
        
        # 2. HTML presence tag (Many phishing emails are entirely HTML to obfuscate text)
        if re.search(r'<\s*a\s+[^>]*href', body_str, re.IGNORECASE) or '<html' in body_str.lower():
            body_has_html_list.append(1)
        else:
            body_has_html_list.append(0)
            
    df_feat['body_length'] = body_length_list
    df_feat['body_has_html'] = body_has_html_list
    
    # 3. TF-IDF
    print("Fitting TF-IDF Vectorizer...")
    # Limiting max_features to 100 to keep the dataset size manageable as tabular data, but big enough for classical ML
    if not os.path.exists(vectorizer_path):
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(body)
        
        # Save vectorizer for Phase 7 (CLI inference)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Saved TF-IDF Vectorizer to {vectorizer_path}")
    else:
        # Load the existing vectorizer for live inference
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = vectorizer.transform(body)
    
    # Convert TF-IDF sparse matrix to DataFrame columns
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()], index=df.index)
    
    return pd.concat([df_feat, tfidf_df], axis=1)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "combined.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "data", "features", "features_matrix.csv")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "trained_models", "tfidf_vectorizer.pkl")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please run data_prep.py first.")
        return
        
    print(f"Loading data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Run the 3-Layer Feature Extractors
    df_headers = extract_header_features(df)
    df_body = extract_body_features(df, VECTORIZER_PATH)
    df_urls = extract_url_features(df)
    
    # Combine all engineered features and the final label
    print("Aggregating all extracted layers into unified matrix...")
    final_df = pd.concat([df_headers, df_body, df_urls, df['label']], axis=1)
    
    # Export to disk
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n======================================")
    print(f"Feature Extraction Complete.")
    print(f"Original shape: {df.shape}")
    print(f"Final Features shape: {final_df.shape}")
    print(f"Exported to: {OUTPUT_CSV}")
    print(f"======================================")

if __name__ == "__main__":
    main()