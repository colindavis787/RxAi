import streamlit as st
import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Drop Claims Table")
st.warning("This will delete all data in the claims table. Proceed with caution.")
if st.button("Drop Claims Table"):
    try:
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        conn = psycopg.connect(url)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS claims")
        conn.commit()
        conn.close()
        st.success("Claims table dropped successfully.")
    except Exception as e:
        st.error(f"Error dropping claims table: {str(e)}")
