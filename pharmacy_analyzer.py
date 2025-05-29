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
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress ARIMA convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Store claims in SQLite database
def store_claims(df, file_name):
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
            cursor.execute('''
                INSERT INTO claims (upload_id, upload_date, column_name, column_value, row_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (upload_id, upload_date, col, str(row[col]), row_id))
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
        if len(member_data) < 5:  # Fallback to linear regression for less data
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
        # Predict utilization with ARIMA
        try:
            y_quantity = member_data[quantity_col].values
            model = ARIMA(y_quantity, order=(1, 0, 0))  # Adjusted to ARIMA(1,0,0)
            model_fit = model.fit()
            predicted_quantities = model_fit.forecast(steps=3)
            predictions[f"{key}_utilization"] = predicted_quantities.tolist()
        except:
            predictions[f"{key}_utilization"] = [0, 0, 0]
        
        # Predict cost with ARIMA
        if cost_cols:
            try:
                y_cost = member_data[cost_cols[0]].values
                model = ARIMA(y_cost, order=(1, 0, 0))
                model_fit = model.fit()
                predicted_costs = model_fit.forecast(steps=3)
                predicted_costs *= (1 + inflation_rate / 4)
                predictions[f"{key}_cost"] = predicted_costs.tolist()
            except:
                predictions[f"{key}_cost"] = [0, 0, 0]
    
    return predictions, f"Predictions generated for next 3 months with {inflation_rate*100}% annual inflation."

# Step 7: Main function
def main(file_path, inflation_rate=0.05):
    df, load_msg = load_claims_file(file_path)
    if df is None:
        return None, load_msg, None, None, None, None
    df, clean_msg = clean_claims_data(df)
    upload_id = store_claims(df, os.path.basename(file_path))
    analysis_results = analyze_claims(df)
    anomalies, anomaly_msg = detect_anomalies(df)
    id_cols = [col for col in df.columns if 'member' in col.lower() or 'id' in col.lower()]
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'service' in col.lower()]
    quantity_cols = [col for col in df.columns if 'quantity' in col.lower()]
    cost_cols = [col for col in df.columns if 'cost' in col.lower()]
    predictions, prediction_msg = predict_utilization_cost(df, id_cols, date_cols, quantity_cols, cost_cols, inflation_rate)
    chart_files = visualize_data(analysis_results, df)
    df.to_csv('cleaned_pharmacy_claims.csv', index=False)
    if not anomalies.empty:
        anomalies.to_csv('anomalies.csv', index=False)
    return df, f"{load_msg}\n{clean_msg}\n{anomaly_msg}\n{prediction_msg}", analysis_results, anomalies, chart_files, predictions

if __name__ == "__main__":
    main('pharmacy_claims.xlsx')