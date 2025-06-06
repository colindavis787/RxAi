# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import datetime
import sqlite3
import numpy as np
import os
import requests
import warnings
from cryptography.fernet import Fernet
import base64

# Suppress ARIMA convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Generate or load encryption key
def get_encryption_key():
    key_file = "encryption_key.key"
    if not os.path.exists(key_file):
        key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(key)
        os.chmod(key_file, 0o600)
    with open(key_file, "rb") as f:
        key = f.read()
    return Fernet(key)

# Encrypt data
def encrypt_data(data, cipher):
    if isinstance(data, str):
        return cipher.encrypt(data.encode()).decode()
    return str(data)  # Convert non-string data to string

# Decrypt data (for display, if needed)
def decrypt_data(data, cipher):
    try:
        return cipher.decrypt(data.encode()).decode()
    except:
        return data

# Fallback medication-to-condition mapping
MEDICATION_CONDITIONS = {
    "AMOXICILLIN": ["Bacterial Infections (e.g., strep throat, ear infections)"],
    "DROSPIRENONE/ETHINYL ESTRADIOL": ["Contraception", "Acne", "Premenstrual Syndrome"],
    "TREMFYA": ["Psoriasis", "Psoriatic Arthritis"],
    "UNKNOWN": ["Unknown Condition"]
}

# Fetch conditions from openFDA API
def get_drug_conditions(drug_name):
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}+OR+openfda.generic_name:{drug_name}&limit=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                indications = data["results"][0].get("indications_and_usage", ["No Conditions Found"])
                return indications if indications else ["No Conditions Found"]
            return ["Drug Not Found"]
        return ["API Error"]
    except:
        return MEDICATION_CONDITIONS.get(drug_name.upper(), MEDICATION_CONDITIONS["UNKNOWN"])

# Store claims in SQLite database with encryption
def store_claims(df, file_name):
    cipher = get_encryption_key()
    conn = sqlite3.connect('claims_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            upload_id TEXT,
            upload_date TEXT,
            column_name TEXT,
            column_value TEXT,
            row_id INTEGER
        )
    ''')
    upload_id = f"{file_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    upload_date = datetime.datetime.now().isoformat()
    for row_id, row in df.iterrows():
        for col in df.columns:
            encrypted_value = encrypt_data(str(row[col]), cipher)
            cursor.execute('''
                INSERT INTO claims (upload_id, upload_date, column_name, column_value, row_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (upload_id, upload_date, col, encrypted_value, row_id))
    conn.commit()
    conn.close()
    return upload_id
            
# Step 1: Load the Excel file
def load_claims_file(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        return df, f"File loaded successfully. Number of rows: {len(df)}, Columns: {list(df.columns)}"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

# Step 2: Clean the data
def clean_claims_data(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if 'date' in col.lower() or 'service' in col.lower():
            try:
                df_clean[col] = pd.to_datetime(
                    df_clean[col], origin='1899-12-30', unit='D', errors='coerce'
                ) if df_clean[col].dtype in ['float', 'int'] else pd.to_datetime(df_clean[col], errors='coerce')
            except:
                pass
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna('Unknown')
    df_clean = df_clean.drop_duplicates()
    return df_clean, f"Data cleaned. Number of rows: {len(df_clean)}"
        
# Step 3: Analyze the data
def analyze_claims(df):
    results = {}
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_cols.size > 0:
        results['numeric_summary'] = df[numeric_cols].agg(['mean', 'sum', 'min', 'max']).round(2)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        results[f'{col}_counts'] = df[col].value_counts().head(10)

    id_cols = [col for col in df.columns if 'member' in col.lower() or 'id' in col.lower()]
    if id_cols:
        results['claims_per_id'] = df.groupby(id_cols[0]).size().reset_index(name='Claim Count')
    
    cost_cols = [col for col in df.columns if 'cost' in col.lower()]
    if cost_cols:
        results['cost_summary'] = df.groupby(id_cols[0])[cost_cols[0]].sum().reset_index(name='Total Cost')

    pharmacy_cols = [col for col in df.columns if 'pharmacy' in col.lower()]
    if pharmacy_cols:
        results['pharmacy_counts'] = df[pharmacy_cols[0]].value_counts().head(10)
            
    # Medication-condition analysis
    drug_cols = [col for col in df.columns if 'drug' in col.lower() or 'medication' in col.lower()]
    if id_cols and drug_cols:
        id_col = id_cols[0]
        drug_col = drug_cols[0]
        member_meds = df.groupby(id_col)[drug_col].unique().reset_index()
        member_meds['Conditions'] = member_meds[drug_col].apply(
            lambda drugs: [
                condition
                for drug in drugs
                for condition in get_drug_conditions(drug)
            ]
        )
        results['member_medications'] = member_meds[[id_col, drug_col, 'Conditions']]
    
    return results
    
# Step 4: Detect anomalies
def detect_anomalies(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_cols.size < 2:   
        return pd.DataFrame(), "No sufficient numeric columns for anomaly detection."
    X = df[numeric_cols].dropna()
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(X)
    anomalies = df[df['Anomaly'] == -1][df.columns]
    return anomalies, f"Detected {len(anomalies)} anomalies."
    
# Step 5: Create charts
def visualize_data(analysis_results, df):
    chart_files = []   
    if 'claims_per_id' in analysis_results:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=analysis_results['claims_per_id'],
            x=analysis_results['claims_per_id'].columns[0],
            y='Claim Count',
            color='#1f77b4'
        )
        plt.title('Claims per ID')
        plt.xlabel('ID')
        plt.ylabel('Claim Count')
        plt.savefig('claims_per_id_bar.png')
        plt.close()
        chart_files.append('claims_per_id_bar.png')
    
    categorical_results = {k: v for k, v in analysis_results.items() if k.endswith('_counts')}
    if categorical_results:
        key = list(categorical_results.keys())[0]
        plt.figure(figsize=(6, 6))
        categorical_results[key].plot.pie(autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title(f'Distribution of {key.replace("_counts", "")}')
        plt.ylabel('')
        plt.savefig('categorical_pie.png')
        plt.close()
        chart_files.append('categorical_pie.png')
        
    return chart_files
                    
# Step 6: Predict future utilization and cost
def predict_utilization_cost(df, id_cols, date_cols, quantity_cols, cost_cols, inflation_rate=0.05):
    """
    Predict total Plan cost for 2026, 2027, 2028 using DATE OF SERVICE and Plan cost.
    Handles interchangeable column names and applies inflation rate.
    Returns a dictionary with years as keys and predicted costs as values, plus a message.
    """
    try:
        # Create a copy to avoid modifying original dataframe
        df = df.copy()
        
        # Find date and cost columns (case-insensitive)
        date_col = None
        cost_col = None
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if col_lower in ['date_of_service', 'service_date', 'claim_date']:
                date_col = col
            if col_lower in ['plan_cost', 'cost', 'plan cost']:
                cost_col = col
        
        # Validate required columns
        if date_col is None or cost_col is None:
            missing = []
            if date_col is None:
                missing.append("DATE OF SERVICE")
            if cost_col is None:
                missing.append("Plan cost")
            return {}, f"Missing required columns: {', '.join(missing)}"
        
        # Convert DATE OF SERVICE to datetime
        try:
            df[date_col] = pd.to_datetime(df[date_col], origin='1899-12-30', unit='D', errors='coerce')
        except:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        if df[date_col].isna().all():
            return {}, "All DATE OF SERVICE values are invalid. Please check the date format."
        
        # Ensure Plan cost is numeric
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
        if df[cost_col].isna().all():
            return {}, "All Plan cost values are invalid. Please check the cost data."
        
        # Drop rows with missing dates or costs
        df = df.dropna(subset=[date_col, cost_col])
        
        # Extract year from date
        df['year'] = df[date_col].dt.year
        
        # Aggregate costs by year
        yearly_costs = df.groupby('year')[cost_col].sum().reset_index()
        
        # Check if we have enough data
        if len(yearly_costs) < 2:
            return {}, "Not enough yearly data for prediction. Need at least 2 years."
        
        # Features: year; Target: total cost
        X = yearly_costs[['year']].values
        y = yearly_costs[cost_col].values
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for 2026, 2027, 2028
        future_years = np.array([[2026], [2027], [2028]])
        predictions = model.predict(future_years)
        
        # Apply inflation rate (compounded annually)
        for i in range(len(predictions)):
            predictions[i] *= (1 + inflation_rate) ** (i + 1)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        # Return dictionary of predictions
        predictions_dict = {
            '2026': predictions[0],
            '2027': predictions[1],
            '2028': predictions[2]
        }
        
        return predictions_dict, f"Predictions generated for 2026–2028 with {inflation_rate*100}% annual inflation."
    
    except Exception as e:
        return {}, f"Error in prediction: {str(e)}"
                
# Step 7: Main function
def main(file_path, inflation_rate=0.05):
    df, load_msg = load_claims_file(file_path)
    if df is None:
        return None, load_msg, None, None, None, None
    df, clean_msg = clean_claims_data(df)
    upload_id = store_claims(df, os.path.basename(file_path))
    analysis_results = analyze_claims(df)
    anomalies, anomaly_msg = detect_anomalies(df)
    cipher = get_encryption_key()
    if not anomalies.empty:
        # Encrypt sensitive columns in anomalies
        sensitive_cols = [col for col in anomalies.columns if 'member' in col.lower() or 'drug' in col.lower()]
        for col in sensitive_cols:
            anomalies[col] = anomalies[col].apply(lambda x: encrypt_data(str(x), cipher))
    id_cols = [col for col in df.columns if 'member' in col.lower() or 'id' in col.lower()]
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'service' in col.lower()]
    quantity_cols = [col for col in df.columns if 'quantity' in col.lower()]
    cost_cols = [col for col in df.columns if 'cost' in col.lower()]
    predictions, prediction_msg = predict_utilization_cost(df, id_cols, date_cols, quantity_cols, cost_cols, inflation_rate)
    chart_files = visualize_data(analysis_results, df)   
    # Encrypt sensitive columns in cleaned_df before saving
    sensitive_cols = [col for col in df.columns if 'member' in col.lower() or 'drug' in col.lower()]
    for col in sensitive_cols:
        df[col] = df[col].apply(lambda x: encrypt_data(str(x), cipher))
    df.to_csv('cleaned_pharmacy_claims.csv', index=False)
    if not anomalies.empty:
        anomalies.to_csv('anomalies.csv', index=False)
    return df, f"{load_msg}\n{clean_msg}\n{anomaly_msg}\n{prediction_msg}", analysis_results, anomalies, chart_files, predictions
                
if __name__ == "__main__":
    main('pharmacy_claims.xlsx')
