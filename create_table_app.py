import streamlit as st
import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Create Claims Table")
if st.button("Create Table"):
    try:
        url = os.getenv('DATABASE_URL')
        if not url:
            st.error("DATABASE_URL environment variable not set")
        conn = psycopg.connect(url)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id SERIAL PRIMARY KEY,
                upload_id TEXT,
                hashed_ssn TEXT,
                date DATE,
                medication TEXT,
                cost FLOAT
            )
        """)
        conn.commit()
        st.success("Claims table created successfully or already exists.")
        
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [row[0] for row in cursor.fetchall()]
        if 'claims' in tables:
            st.write("Verified: 'claims' table exists.")
        else:
            st.error("Error: 'claims' table not found after creation attempt.")
        
        conn.close()
    except Exception as e:
        st.error(f"Error: {str(e)}")
