import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pandas as pd
import numpy as np
import psycopg
import os
from pharmacy_analyzer import hash_ssn

# Define medication lists (example, update with actual lists)
HYPERTENSION_MEDS = ["Lisinopril", "Amlodipine"]
KIDNEY_DISEASE_MEDS = ["Furosemide", "Spironolactone"]

# Fetch historical data from database
def fetch_claims_data():
    try:
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        conn = psycopg.connect(dbname=url.split('/')[3],
                              user=url.split('//')[1].split(':')[0],
                              password=url.split('//')[1].split(':')[1].split('@')[0],
                              host=url.split('@')[1].split(':')[0],
                              port=url.split(':')[3].split('/')[0],
                              sslmode='require')
        df = pd.read_sql("SELECT hashed_ssn, date, medication, cost FROM claims ORDER BY hashed_ssn, date", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching claims data: {str(e)}")
        return pd.DataFrame()

# Prepare sequences for LSTM
def prepare_sequences(df, seq_length=5):
    all_meds = df['medication'].unique()
    med_to_idx = {med: idx + 1 for idx, med in enumerate(all_meds)}  # 0 for padding
    sequences = []
    labels = []
    ssns = df['hashed_ssn'].unique()
    for ssn in ssns:
        ssn_df = df[df['hashed_ssn'] == ssn].sort_values('date')
        meds = ssn_df['medication'].tolist()
        if len(meds) >= seq_length:
            for i in range(len(meds) - seq_length):
                seq = meds[i:i + seq_length]
                next_med = meds[i + seq_length] if i + seq_length < len(meds) else None
                if next_med and any(m in HYPERTENSION_MEDS for m in seq):
                    sequences.append([med_to_idx[m] for m in seq])
                    labels.append(1 if next_med in KIDNEY_DISEASE_MEDS else 0)
    return np.array(sequences), np.array(labels), med_to_idx

# Define LSTM model
def build_model(vocab_size, seq_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_length),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Predict for new data
def predict_future_meds(df, model, med_to_idx, seq_length=5):
    predictions = {}
    ssns = df['hashed_ssn'].unique()
    for ssn in ssns:
        ssn_df = df[df['hashed_ssn'] == ssn].sort_values('date')
        if len(ssn_df) >= seq_length:
            seq = ssn_df['medication'].iloc[-seq_length:].tolist()
            if any(m in HYPERTENSION_MEDS for m in seq):
                seq_idx = [med_to_idx.get(m, 0) for m in seq]
                prob = model.predict(np.array([seq_idx]), verbose=0)[0][0]
                predictions[ssn] = prob
    return predictions

if __name__ == "__main__":
    df = fetch_claims_data()
    if not df.empty:
        X, y, med_to_idx = prepare_sequences(df)
        if len(X) > 0:
            model = build_model(len(med_to_idx) + 1, 5)
            model.fit(X, y, epochs=10, batch_size=32)
            model.save('lstm_model.h5')
            print("Model trained and saved as lstm_model.h5")
        else:
            print("Not enough data to train the model yet.")
    else:
        print("No claims data found in database.")
