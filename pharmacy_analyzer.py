# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import datetime

# Step 1: Load the Excel file
def load_claims_file(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        return df, f"File loaded successfully. Number of rows: {len(df)}"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

# Step 2: Clean the data
def clean_claims_data(df):
    df_clean = df.copy()
    # Convert DATE OF SERVICE to datetime
    df_clean['DATE OF SERVICE'] = pd.to_datetime(
        df_clean['DATE OF SERVICE'], origin='1899-12-30', unit='D', errors='coerce'
    )
    # Fill missing values
    df_clean = df_clean.fillna({
        'QUANTITY': 0,
        'DAYS SUPPLY': 0,
        'Specialty Indicator': 'N'
    })
    # Ensure correct data types
    df_clean = df_clean.astype({
        'Member Number': int,
        'QUANTITY': float,
        'DAYS SUPPLY': int,
        'NDC': str
    }, errors='ignore')
    # Remove duplicates and negative quantities
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean[df_clean['QUANTITY'] >= 0]
    return df_clean, f"Data cleaned. Number of rows: {len(df_clean)}"

# Step 3: Analyze the data
def analyze_claims(df):
    # Claims per member
    claims_per_member = df.groupby('Member Number').size().reset_index(name='Claim Count')
    # Brand vs generic
    brand_generic_dist = df['BRAND/GENERIC INDICATOR'].value_counts()
    # Channel
    channel_dist = df['Channel'].value_counts()
    # Drug stats
    drug_stats = df.groupby('Drug Name').agg({
        'QUANTITY': 'mean',
        'DAYS SUPPLY': 'mean'
    }).reset_index()
    return {
        'claims_per_member': claims_per_member,
        'brand_generic_dist': brand_generic_dist,
        'channel_dist': channel_dist,
        'drug_stats': drug_stats
    }

# Step 4: Detect anomalies
def detect_anomalies(df):
    features = ['QUANTITY', 'DAYS SUPPLY']
    X = df[features].dropna()
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(X)
    anomalies = df[df['Anomaly'] == -1][[
        'Member Number', 'Drug Name', 'QUANTITY', 'DAYS SUPPLY', 'DATE OF SERVICE'
    ]]
    return anomalies

# Step 5: Create charts
def visualize_data(analysis_results):
    # Pie chart
    plt.figure(figsize=(6, 6))
    analysis_results['brand_generic_dist'].plot.pie(
        autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e']
    )
    plt.title('Brand vs Generic Distribution')
    plt.ylabel('')
    plt.savefig('brand_generic_pie.png')
    plt.close()
    # Bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=analysis_results['claims_per_member'],
        x='Member Number',
        y='Claim Count',
        color='#1f77b4'
    )
    plt.title('Claims per Member')
    plt.xlabel('Member Number')
    plt.ylabel('Claim Count')
    plt.savefig('claims_per_member_bar.png')
    plt.close()
    return ['brand_generic_pie.png', 'claims_per_member_bar.png']

# Step 6: Main function
def main(file_path):
    # Load
    df, load_msg = load_claims_file(file_path)
    if df is None:
        return None, load_msg, None, None, None
    # Clean
    df, clean_msg = clean_claims_data(df)
    # Analyze
    analysis_results = analyze_claims(df)
    # Anomalies
    anomalies = detect_anomalies(df)
    # Charts
    chart_files = visualize_data(analysis_results)
    # Save results
    df.to_csv('cleaned_pharmacy_claims.csv', index=False)
    anomalies.to_csv('anomalies.csv', index=False)
    return df, f"{load_msg}\n{clean_msg}", analysis_results, anomalies, chart_files

if __name__ == "__main__":
    main('pharmacy_claims.xlsx')
