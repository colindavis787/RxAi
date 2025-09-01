# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import datetime
import numpy as np
import os
import requests
import warnings
from cryptography.fernet import Fernet
import base64
import boto3
import psycopg
from urllib.parse import urlparse

# Suppress ARIMA convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = os.getenv('S3_BUCKET', 'rxai-phi-854611169949')

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

# Database connection
def get_db_connection():
    try:
        url = urlparse(os.getenv('DATABASE_URL'))
        conninfo = {
            'dbname': url.path[1:],
            'user': url.username,
            'password': url.password,
            'host': url.hostname,
            'port': url.port,
            'sslmode': 'verify-full',
            'sslrootcert': '/app/us-east-1-bundle.pem'  # Elastic Beanstalk path
        }
        if os.getenv('AWS_EXECUTION_ENV') is None:
            conninfo['sslrootcert'] = '/Users/colindavis/Desktop/pharmacy_claims_analyzer/us-east-1-bundle.pem'
        return psycopg.connect(**conninfo)
    except Exception as e:
        print(f"Failed to connect to database: {str(e)}")
        raise

# Generate or load encryption key
def get_encryption_key():
    key_file = "encryption_key.key"
    temp_key_file = "/tmp/encryption_key.key"
    if not os.path.exists(temp_key_file):
        try:
            s3.download_file(bucket_name, key_file, temp_key_file)
        except s3.exceptions.NoSuchKey:
            key = Fernet.generate_key()
            with open(temp_key_file, "wb") as f:
                f.write(key)
            s3.upload_file(temp_key_file, bucket_name, key_file)
        os.chmod(temp_key_file, 0o600)
    with open(temp_key_file, "rb") as f:
        key = f.read()
    return Fernet(key)

# Encrypt data
def encrypt_data(data, cipher):
    if isinstance(data, str):
        return cipher.encrypt(data.encode()).decode()
    return str(data)

# Decrypt data
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

# Store claims in RDS database with encryption
def store_claims(df, file_name):
    cipher = get_encryption_key()
    upload_id = f"{file_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    upload_date = datetime.datetime.now().isoformat()
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            for row_id, row in df.iterrows():
                for col in df.columns:
                    encrypted_value = encrypt_data(str(row[col]), cipher)
                    cursor.execute(
                        '''
                        INSERT INTO claims (upload_id, upload_date, column_name, column_value, row_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ''',
                        (upload_id, upload_date, col, encrypted_value, row_id)
                    )
        conn.commit()
        conn.close()
        return upload_id
    except Exception as e:
        print(f"Error storing claims: {str(e)}")
        return None

# Load the Excel file from S3
def load_claims_file(file_path):
    try:
        temp_file = "/tmp/" + os.path.basename(file_path)
        s3.download_file(bucket_name, file_path, temp_file)
        df = pd.read_excel(temp_file, sheet_name=0)
        os.remove(temp_file)
        return df, f"File loaded successfully. Number of rows: {len(df)}, Columns: {list(df.columns)}"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

# Clean the data
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

# Analyze the data
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
    
    id_cols = [col for col in df.columns if 'member' in col.lower() or 'id' in col.lower()]
    if id_cols:
        results['claims_per_id'] = df.groupby(id_cols[0]).size().reset_index(name='Claim Count')
    
    cost_cols = [col for col in df.columns if 'cost' in col.lower()]
    if cost_cols:
        results['cost_summary'] = df.groupby(id_cols[0])[cost_cols[0]].sum().reset_index(name='Total Cost')
    
    pharmacy_cols = [col for col in df.columns if 'pharmacy' in col.lower()]
    if pharmacy_cols:
        results['pharmacy_counts'] = df[pharmacy_cols[0]].value_counts().head(10)
    
    drug_cols = [col for col in df.columns if 'drug' in col.lower() or 'medication' in col.lower()]
    if id_cols and drug_cols:
        id_col = id_cols[0]
        drug_col = drug_cols[0]
        member_meds = df.groupby(id_col)[drug_col].unique().reset_index()
        member_meds['Conditions'] = member_meds[drug_col].apply(
            lambda drugs: [condition for drug in drugs for condition in get_drug_conditions(drug)]
        )
        results['member_medications'] = member_meds[[id_col, drug_col, 'Conditions']]
    
    return results

# Detect anomalies
def detect_anomalies(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_cols.size < 2:
        return pd.DataFrame(), "No sufficient numeric columns for anomaly detection."
    X = df[numeric_cols].dropna()
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(X)
    anomalies = df[df['Anomaly'] == -1][df.columns]
    return anomalies, f"Detected {len(anomalies)} anomalies."

# Create charts and upload to S3
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
        temp_chart = "/tmp/claims_per_id_bar.png"
        plt.savefig(temp_chart)
        plt.close()
        s3.upload_file(temp_chart, bucket_name, "charts/claims_per_id_bar.png")
        chart_files.append("charts/claims_per_id_bar.png")
        os.remove(temp_chart)
    
    categorical_results = {k: v for k, v in analysis_results.items() if k.endswith('_counts')}
    if categorical_results:
        key = list(categorical_results.keys())[0]
        plt.figure(figsize=(6, 6))
        categorical_results[key].plot.pie(autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title(f'Distribution of {key.replace("_counts", "")}')
        plt.ylabel('')
        temp_chart = "/tmp/categorical_pie.png"
        plt.savefig(temp_chart)
        plt.close()
        s3.upload_file(temp_chart, bucket_name, "charts/categorical_pie.png")
        chart_files.append("charts/categorical_pie.png")
        os.remove(temp_chart)
    
    return chart_files

# Predict future utilization and cost
def predict_utilization_cost(df, id_cols, date_cols, quantity_cols, cost_cols, inflation_rate=0.05):
    predictions = {}
    if not (id_cols and date_cols and quantity_cols):
        return predictions, "Missing required columns for prediction."
    
    id_col = id_cols[0]
    date_col = date_cols[0]
    quantity_col = quantity_cols[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    min_date = df[date_col].min()
    df['days_since_min'] = (df[date_col] - min_date).dt.days
    
    df['month'] = df[date_col].dt.to_period('M')
    monthly_data = df.groupby([id_col, 'month']).agg({
        quantity_col: 'sum',
        cost_cols[0]: 'sum' if cost_cols else lambda x: 0
    }).reset_index()
    monthly_data['month_ordinal'] = monthly_data['month'].apply(lambda x: x.ordinal)
    
    for key, value in monthly_data.groupby(id_col):
        member_data = value.sort_values('month_ordinal')
        if len(member_data) < 5:
            if len(member_data) >= 2:
                X = member_data['month_ordinal'].values.reshape(-1, 1)
                y_quantity = member_data[quantity_col].values
                model = LinearRegression()
                model.fit(X, y_quantity)
                future_months = np.array([X[-1][0] + i for i in range(1, 4)]).reshape(-1, 1)
                predicted_quantities = model.predict(future_months)
                predictions[f"{key}_utilization"] = predicted_quantities.tolist()
                
                if cost_cols:
                    y_cost = member_data[cost_cols[0]].values
                    model.fit(X, y_cost)
                    predicted_costs = model.predict(future_months)
                    predicted_costs *= (1 + inflation_rate / 4)
                    predictions[f"{key}_cost"] = predicted_costs.tolist()
            continue
        try:
            y_quantity = member_data[quantity_col].values
            model = LinearRegression()
            X = member_data['month_ordinal'].values.reshape(-1, 1)
            model.fit(X, y_quantity)
            future_months = np.array([X[-1][0] + i for i in range(1, 4)]).reshape(-1, 1)
            predicted_quantities = model.predict(future_months)
            predictions[f"{key}_utilization"] = predicted_quantities.tolist()
        except:
            predictions[f"{key}_utilization"] = [0, 0, 0]
        
        if cost_cols:
            try:
                y_cost = member_data[cost_cols[0]].values
                model = LinearRegression()
                model.fit(X, y_cost)
                predicted_costs = model.predict(future_months)
                predicted_costs *= (1 + inflation_rate / 4)
                predictions[f"{key}_cost"] = predicted_costs.tolist()
            except:
                predictions[f"{key}_cost"] = [0, 0, 0]
    
    return predictions, f"Predictions generated for next 3 months with {inflation_rate*100}% annual inflation."

# Main function
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
        sensitive_cols = [col for col in anomalies.columns if 'member' in col.lower() or 'drug' in col.lower()]
        for col in sensitive_cols:
            anomalies[col] = anomalies[col].apply(lambda x: encrypt_data(str(x), cipher))
    id_cols = [col for col in df.columns if 'member' in col.lower() or 'id' in col.lower()]
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'service' in col.lower()]
    quantity_cols = [col for col in df.columns if 'quantity' in col.lower()]
    cost_cols = [col for col in df.columns if 'cost' in col.lower()]
    predictions, prediction_msg = predict_utilization_cost(df, id_cols, date_cols, quantity_cols, cost_cols, inflation_rate)
    chart_files = visualize_data(analysis_results, df)
    sensitive_cols = [col for col in df.columns if 'member' in col.lower() or 'drug' in col.lower()]
    for col in sensitive_cols:
        df[col] = df[col].apply(lambda x: encrypt_data(str(x), cipher))
    temp_csv = "/tmp/cleaned_pharmacy_claims.csv"
    df.to_csv(temp_csv, index=False)
    s3.upload_file(temp_csv, bucket_name, "csv/cleaned_pharmacy_claims.csv")
    os.remove(temp_csv)
    if not anomalies.empty:
        temp_csv = "/tmp/anomalies.csv"
        anomalies.to_csv(temp_csv, index=False)
        s3.upload_file(temp_csv, bucket_name, "csv/anomalies.csv")
        os.remove(temp_csv)
    return df, f"{load_msg}\n{clean_msg}\n{anomaly_msg}\n{prediction_msg}", analysis_results, anomalies, chart_files, predictions

if __name__ == "__main__":
    main('pharmacy_claims.xlsx')
