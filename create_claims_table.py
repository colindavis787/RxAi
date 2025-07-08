import psycopg
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to the database and create the claims table
try:
    url = os.getenv('DATABASE_URL')
    if not url:
        raise ValueError("DATABASE_URL environment variable not set")
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
    print("Claims table created successfully or already exists.")
    
    # Verify table existence
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [row[0] for row in cursor.fetchall()]
    if 'claims' in tables:
        print("Verified: 'claims' table exists.")
    else:
        print("Error: 'claims' table not found after creation attempt.")
    
    conn.close()
except Exception as e:
    print(f"Error: {str(e)}")
