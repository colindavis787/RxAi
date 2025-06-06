import sqlite3

# Connect to the database (creates it if it doesn't exist)
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Create the users table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        password TEXT NOT NULL
    )
''')

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database initialized successfully with users table.")
