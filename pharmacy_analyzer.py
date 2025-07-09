import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from openai import OpenAI
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# Hash SSN for privacy
def hash_ssn(ssn):
    try:
        salt = os.getenv('SALT', 'default_salt_1234567890').encode('utf-8')
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(str(ssn).encode('utf-8')))
        return key.decode('utf-8')
    except Exception as e:
        print(f"Error hashing SSN: {str(e)}")
        return str(ssn)

# Main analysis function
def main(file_path, inflation_rate):
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        messages = ["File read successfully."]

        # Required columns (SSN is optional)
        required_columns = ['Date of Service', 'Drug Name', 'Plan Cost']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            messages.append(f"Missing required columns: {', '.join(missing)}. Analysis aborted.")
            return None, messages, {}, pd.DataFrame(), [], {}

        # Convert Date of Service to datetime
        df['Date of Service'] = pd.to_datetime(df['Date of Service'], errors='coerce')
        messages.append("Converted Date of Service to datetime.")

        # Handle missing values
        df = df.dropna(subset=required_columns)
        messages.append(f"Dropped {len(df) - len(df.dropna())} rows with missing values.")

        # Hash SSNs if present
        has_ssn = any(col.lower() in ['ssn', 'social security number'] for col in df.columns)
        if has_ssn:
            ssn_col = next(col for col in df.columns if col.lower() in ['ssn', 'social security number'])
            df['Hashed SSN'] = df[ssn_col].apply(hash_ssn)
            messages.append("Hashed SSNs for privacy.")
        else:
            df['Hashed SSN'] = 'N/A'
            messages.append("No SSN column found; skipping SSN hashing.")

        # Basic analysis
        analysis_results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        analysis_results['numeric_summary'] = df[numeric_cols].describe()
        messages.append("Generated numeric summary.")

        # Categorical analysis
        categorical_cols = ['Drug Name']
        for col in categorical_cols:
            analysis_results[f"{col}_counts"] = df[col].value_counts()

        # Claims per ID (if SSN present)
        if has_ssn:
            analysis_results['claims_per_id'] = df.groupby('Hashed SSN').size().reset_index(name='Claim Count')
            messages.append("Calculated claims per ID.")
        else:
            analysis_results['claims_per_id'] = pd.DataFrame({'Hashed SSN': ['N/A'], 'Claim Count': [len(df)]})
            messages.append("No SSN column; counted total claims.")

        # Cost summary
        if has_ssn:
            analysis_results['cost_summary'] = df.groupby('Hashed SSN')['Plan Cost'].sum().reset_index(name='Total Cost')
        else:
            analysis_results['cost_summary'] = pd.DataFrame({'Hashed SSN': ['N/A'], 'Total Cost': [df['Plan Cost'].sum()]})
        messages.append("Calculated total cost.")

        # Anomaly detection
        iso = IsolationForest(contamination=0.1, random_state=42)
        df_numeric = df[numeric_cols]
        anomalies = df.copy()
        anomalies['Anomaly'] = iso.fit_predict(df_numeric)
        anomalies = anomalies[anomalies['Anomaly'] == -1][['Hashed SSN', 'Date of Service', 'Drug Name', 'Plan Cost']]
        messages.append("Performed anomaly detection.")

        # Visualizations
        chart_files = []
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Plan Cost'], bins=30)
        plt.title('Distribution of Plan Costs')
        cost_hist_path = 'cost_histogram.png'
        plt.savefig(cost_hist_path)
        plt.close()
        chart_files.append(cost_hist_path)
        messages.append("Generated cost histogram.")

        plt.figure(figsize=(10, 6))
        sns.countplot(x='Drug Name', data=df)
        plt.title('Drug Name Distribution')
        plt.xticks(rotation=45)
        drug_counts_path = 'drug_counts.png'
        plt.savefig(drug_counts_path)
        plt.close()
        chart_files.append(drug_counts_path)
        messages.append("Generated drug counts plot.")

        # Cost trend prediction
        predictions = {}
        current_year = df['Date of Service'].dt.year.min()
        if pd.notna(current_year):
            total_cost = df['Plan Cost'].sum()
            for i in range(4):
                year = current_year + i
                predictions[str(year)] = total_cost * (1 + inflation_rate) ** i
            messages.append("Generated cost predictions with inflation.")
        else:
            messages.append("No valid dates for cost predictions.")

        # Medication and condition inference
        conditions = []
        for _, row in df.iterrows():
            drug = row['Drug Name']
            try:
                response = client.chat.completions.create(
                    model="grok-3",
                    messages=[
                        {"role": "system", "content": "Identify medical conditions associated with a given medication."},
                        {"role": "user", "content": f"What conditions is {drug} used to treat?"}
                    ],
                    max_tokens=100
                )
                condition = response.choices[0].message.content.strip()
            except Exception as e:
                condition = f"Error: {str(e)}"
            conditions.append(condition)
        df['Conditions'] = conditions
        analysis_results['member_medications'] = df[['Hashed SSN', 'Drug Name', 'Conditions']].drop_duplicates()
        messages.append("Inferred medical conditions for medications.")

        return df, messages, analysis_results, anomalies, chart_files, predictions
    except Exception as e:
        messages.append(f"Error processing file: {str(e)}")
        return None, messages, {}, pd.DataFrame(), [], {}
