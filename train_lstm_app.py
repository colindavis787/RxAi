import streamlit as st
from lstm_model import fetch_claims_data, prepare_sequences, build_model
from dotenv import load_dotenv

load_dotenv()

st.title("Train LSTM Model")
if st.button("Train Model"):
    df = fetch_claims_data()
    if not df.empty:
        X, y, med_to_idx, ndc_to_idx = prepare_sequences(df)
        if len(X) > 0:
            model = build_model(len(med_to_idx) + 1, len(ndc_to_idx) + 1, 5)
            model.fit(X, y, epochs=10, batch_size=32)
            model.save('lstm_model.h5')
            st.success("Model trained and saved as lstm_model.h5")
        else:
            st.error("Not enough data to train the model yet.")
    else:
        st.error("No claims data found in database.")
