import pandas as pd
import os

def prepare_data(nazario_path, spamassassin_path, output_path):
    print("Starting data preparation...")
    
    # 1. Load Nazario dataset (Contains only phishing)
    try:
        nz_df = pd.read_csv(nazario_path)
        nz_df['label'] = 1
        print(f"Loaded {len(nz_df)} phishing samples from Nazario.")
    except Exception as e:
        print(f"Error loading {nazario_path}: {e}")
        return

    # 2. Load SpamAssassin dataset and filter for legitimate (ham)
    try:
        sa_df = pd.read_csv(spamassassin_path)
        # Assuming label '0' is "0" as string or int 0 based on PowerShell output format
        # To be safe, cast to string then filter, then assign int 0
        sa_df = sa_df[sa_df['label'].astype(str) == '0'].copy()
        sa_df['label'] = 0
        print(f"Loaded {len(sa_df)} legitimate samples from SpamAssassin.")
    except Exception as e:
        print(f"Error loading {spamassassin_path}: {e}")
        return

    # 3. Standardize and concatenate
    # pandas concat will automatically align columns by header name
    print("Merging datasets into a single balanced corpus...")
    combined_df = pd.concat([nz_df, sa_df], ignore_index=True)
    
    # Extract only the columns we need across our 3 layers
    columns_to_keep = ['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label']
    combined_df = combined_df[[col for col in columns_to_keep if col in combined_df.columns]]
    
    # 4. Shuffle the data so training sets are distributed randomly
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"\n======================================")
    print(f"Total Combined Records: {len(combined_df)}")
    print(f"Target distribution:\n{combined_df['label'].value_counts()}")
    print(f"Exported combined dataset successfully to {output_path}.")
    print(f"======================================")

if __name__ == "__main__":
    # Ensure absolute/relative resolution works by anchoring to script directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    NAZARIO_CSV = os.path.join(BASE_DIR, "data", "Nazario.csv")
    SPAMASSASSIN_CSV = os.path.join(BASE_DIR, "data", "SpamAssasin.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "combined.csv")
    
    prepare_data(NAZARIO_CSV, SPAMASSASSIN_CSV, OUTPUT_CSV)