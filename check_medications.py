import streamlit as st
import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Check Medications in Claims Table")
if st.button("List Medications"):
    try:
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        conn = psycopg.connect(url)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT medication FROM claims")
        medications = [row[0] for row in cursor.fetchall()]
        conn.close()
        if medications:
            st.write("Unique Medications in Claims Table:")
            for med in medications:
                st.write(f"- {med}")
        else:
            st.error("No medications found in claims table.")
    except Exception as e:
        st.error(f"Error fetching medications: {str(e)}")
