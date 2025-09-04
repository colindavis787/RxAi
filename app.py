import streamlit as st
import os
from pharmacy_analyzer import main, hash_ssn
from openai import OpenAI
import pandas as pd
import sqlite3
import yaml
from yaml.loader import SafeLoader
import jwt
import datetime
from pathlib import Path
import logging
from urllib.parse import unquote
import psycopg
import plotly.express as px
from dotenv import load_dotenv
import tensorflow as tf
from lstm_model import predict_future_meds, fetch_claims_data

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Debug prints to verify user and directory (workaround for os.getlogin)
import pwd
user = os.environ.get('USER') or pwd.getpwuid(os.getuid())[0]
print(f"Current user: {user}")
print(f"Current directory: {os.getcwd()}")

# Limit file watching to specific files
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"  # Disable watching (use with caution)

# Certificate copy logic with debug
import shutil
cert_source = "/mount/src/rxai/us-east-1-bundle.pem"
cert_dest = "/home/appuser/.postgresql/root.crt"
if not os.path.exists("/home/appuser/.postgresql"):
    os.makedirs("/home/appuser/.postgresql")
    logger.debug(f"Created directory: /home/appuser/.postgresql")
if os.path.exists(cert_source):
    shutil.copy(cert_source, cert_dest)
    logger.debug(f"Copied certificate from {cert_source} to {cert_dest}")
else:
    logger.error(f"Certificate not found at {cert_source}")

# JWT secret key
JWT_SECRET_KEY = 'your_jwt_secret_key_12345'

# Load user credentials from database
from urllib.parse import urlparse, parse_qs

def load_users():
    try:
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        parsed_url = urlparse(url)
        # Extract database name from path, ignoring query parameters
        dbname = parsed_url.path[1:]  # Remove leading '/' from path
        conninfo = {
            'dbname': dbname,
            'user': parsed_url.username,
            'password': parsed_url.password,
            'host': parsed_url.hostname,
            'port': parsed_url.port,
            'sslmode': parse_qs(parsed_url.query).get('sslmode', ['prefer'])[0]  # Default to 'prefer' if not set
        }
        logger.debug(f"Connecting to Postgres database with conninfo: {conninfo}")
        conn = psycopg.connect(**conninfo)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute("SELECT username, name, password FROM users")
            rows = cursor.fetchall()
            users = {row['username']: {'name': row['name'], 'password': row['password']} for row in rows}
        conn.close()
        logger.debug(f"Loaded users from database: {list(users.keys())}")
        if not users:
            logger.warning("No users found in database")
        return users
    except Exception as e:
        logger.error(f"Failed to load users from database: {str(e)}")
        st.error(f"Database connection failed: {str(e)}. Using empty user list.")
        return {}  # Default to empty dict so app starts

# Create claims table
def create_claims_table():
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
                date_of_service DATE,
                medication TEXT,
                cost FLOAT,
                quantity FLOAT,
                ndc TEXT,
                days_supply INTEGER
            )
        """)
        conn.commit()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        if 'claims' in tables:
            st.success("Claims table created successfully or already exists.")
        else:
            st.error("Error: 'claims' table not found after creation attempt.")
    except Exception as e:
        st.error(f"Error creating claims table: {str(e)}")

# Verify claims table exists
def verify_claims_table():
    try:
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        conn = psycopg.connect(url)
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        if 'claims' in tables:
            st.success("Verified: 'claims' table exists in the database.")
        else:
            st.error("Error: 'claims' table does not exist.")
    except Exception as e:
        st.error(f"Error verifying claims table: {str(e)}")

# Store claims in PostgreSQL database
def store_claims(df, upload_id):
    has_ssn = any(col.lower() in ['ssn', 'social security number'] for col in df.columns)
    if not has_ssn:
        st.warning("No SSN column found; skipping database storage.")
        return
    try:
        url = os.getenv('DATABASE_URL')
        conn = psycopg.connect(dbname=url.split('/')[3],
                             user=url.split('//')[1].split(':')[0],
                             password=url.split('//')[1].split(':')[1].split('@')[0],
                             host=url.split('@')[1].split(':')[0],
                             port=url.split(':')[3].split('/')[0],
                             sslmode='require')
        cursor = conn.cursor()
        ssn_col = next(col for col in df.columns if col.lower() in ['ssn', 'social security number'])
        date_col = [col for col in df.columns if 'date' in col.lower() or 'service' in col.lower()]
        date_col = date_col[0] if date_col else None
        drug_col = [col for col in df.columns if 'drug' in col.lower() or 'medication' in col.lower()]
        drug_col = drug_col[0] if drug_col else None
        cost_col = [col for col in df.columns if 'cost' in col.lower()]
        cost_col = cost_col[0] if cost_col else None
        quantity_col = [col for col in df.columns if 'quantity' in col.lower()]
        quantity_col = quantity_col[0] if quantity_col else None
        ndc_col = [col for col in df.columns if 'ndc' in col.lower()]
        ndc_col = ndc_col[0] if ndc_col else None
        days_supply_col = [col for col in df.columns if 'days supply' in col.lower() or 'days_supply' in col.lower()]
        days_supply_col = days_supply_col[0] if days_supply_col else None
        if not all([ssn_col, date_col, drug_col, cost_col]):
            missing = [c for c, v in [('SSN', ssn_col), ('Date', date_col), ('Drug', drug_col), ('Cost', cost_col)] if not v]
            logger.error(f"Missing required columns: {', '.join(missing)}")
            st.error(f"Missing required columns: {', '.join(missing)}")
            return
        for _, row in df.iterrows():
            cursor.execute(
                "INSERT INTO claims (upload_id, hashed_ssn, date_of_service, medication, cost, quantity, ndc, days_supply) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    upload_id,
                    hash_ssn(row[ssn_col]),
                    row[date_col],
                    row[drug_col],
                    row[cost_col],
                    row[quantity_col] if quantity_col else None,
                    row[ndc_col] if ndc_col else None,
                    int(row[days_supply_col]) if days_supply_col and pd.notnull(row[days_supply_col]) else None
                )
            )
        conn.commit()
        conn.close()
        logger.debug(f"Stored claims for upload_id: {upload_id}")
        st.success(f"Claims stored in database for upload ID: {upload_id}")
    except Exception as e:
        logger.error(f"Failed to store claims: {str(e)}")
        st.error(f"Database error: {str(e)}")

users = load_users()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'name' not in st.session_state:
    st.session_state.name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Get username and token from URL parameters
query_params = st.query_params.to_dict()
logger.debug(f"Raw query parameters: {query_params}")
username = query_params.get('username')
token = query_params.get('token')
embedded = query_params.get('embedded', 'false').lower() == 'true'

token = unquote(token) if token else None
logger.debug(f"Token after decoding: {token}")
if token:
    segments = token.split('.')
    logger.debug(f"Token segments count: {len(segments)}")
if username and token and not st.session_state.authenticated:
    try:
        if token.count('.') != 2:
            logger.error(f"Malformed token received: {token}")
            st.error("Invalid authentication token format.")
        else:
            logger.debug(f"Decoding token for username: {username}, token: {token}")
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            if payload['username'] == username and username in users:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.name = users[username].get('name', 'User')
            else:
                logger.warning("Token username mismatch or user not found")
                st.error("Invalid authentication token.")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token error: {str(e)}")
        st.error("Invalid authentication token.")
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        st.error("Authentication token has expired. Please log in again.")
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {str(e)}")
        st.error("Error validating authentication token.")

# Display the app if authenticated
if st.session_state.authenticated:
    st.markdown(
        """
        <style>
        .main { background-color: #FFFFFF; padding: 20px; border-radius: 10px; }
        .stButton>button { background-color: #003087; color: white; border-radius: 5px; }
        .stDataFrame {
            max-height: 400px;
            overflow-x: auto;
            width: 100%;
            min-width: 1000px;
        }
        .stDataFrame table {
            width: 100%;
            table-layout: auto;
        }
        .stDataFrame th, .stDataFrame td {
            white-space: normal;
            text-align: left;
            word-wrap: break-word;
            max-width: 300px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Pharmacy Claims Analyzer")
    st.markdown(f"Welcome, *{st.session_state['name']}*!", unsafe_allow_html=True)
    st.sidebar.markdown("[Logout](https://rxaianalytics.com/logout)")

    # Admin actions in sidebar
    st.sidebar.header("Admin Actions")
    if st.sidebar.button("Create Claims Table"):
        create_claims_table()
    if st.sidebar.button("Verify Claims Table"):
        verify_claims_table()

    try:
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    except Exception as e:
        st.error(f"Error initializing AI client: {str(e)}")
        client = None

    with st.container():
        st.header("Past Uploads")
        try:
            conn = sqlite3.connect('claims_history.db')
            past_uploads = pd.read_sql("SELECT DISTINCT upload_id, upload_date FROM claims", conn)
            conn.close()
            if not past_uploads.empty:
                st.dataframe(past_uploads.head(25), use_container_width=True, hide_index=True)
            else:
                st.write("No past uploads found.")
        except Exception as e:
            st.error(f"Error accessing historical data: {str(e)}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Upload and Analyze")
        inflation_rate = st.slider(
            "Annual Drug Price Inflation (%)",
            min_value=0.0, max_value=10.0, value=5.0, step=0.1, format="%.1f"
        ) / 100
        uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
    with col2:
        st.header("Quick Stats")
        if uploaded_file:
            st.write("File uploaded. Processing...")

    if uploaded_file:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size exceeds 10MB limit.")
        else:
            temp_file = "temp.xlsx"
            upload_id = f"upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            cleaned_df, messages, analysis_results, anomalies, chart_files, predictions = main(temp_file, inflation_rate)
            if cleaned_df is not None:
                store_claims(cleaned_df, upload_id)
            st.subheader("Processing Status")
            for msg in messages:
                st.write(msg)

            if cleaned_df is not None:
                tabs = st.tabs(["Summary", "Visualizations", "Anomalies", "Predictions", "Q&A"])
                with tabs[0]:
                    st.subheader("Analysis Summary")
                    if 'numeric_summary' in analysis_results:
                        st.write("Numeric Column Summary")
                        numeric_summary = analysis_results['numeric_summary'].drop(columns=['SSN', 'Social Security Number', 'NDC', 'AWP'], errors='ignore')
                        st.dataframe(numeric_summary.head(25), use_container_width=True, hide_index=True)

                    for key, value in analysis_results.items():
                        if key.endswith('_counts') and 'B/G Fill Indicator' not in key:
                            st.write(f"### {key.replace('_counts', '')} Distribution")
                            st.dataframe(value.head(25).reset_index(), use_container_width=True, hide_index=True)
                        elif key == 'claims_per_id':
                            st.write("### Claims per ID")
                            st.dataframe(value.head(25), use_container_width=True, hide_index=True)
                        elif key == 'cost_summary':
                            st.write("### Total Cost per ID")
                            st.dataframe(value.head(25), use_container_width=True, hide_index=True)
                        elif key == 'member_medications':
                            st.write("### Medications and Conditions")
                            value_df = value.copy()
                            value_df['Conditions'] = value_df['Conditions'].apply(lambda x: '; '.join(str(i) for i in x) if isinstance(x, list) else str(x))
                            value_df['Drug Name'] = value_df['Drug Name'].apply(lambda x: '; '.join(str(i) for i in x) if isinstance(x, list) else str(x))
                            st.dataframe(value_df.head(25), use_container_width=True, hide_index=True)

                with tabs[1]:
                    st.subheader("Visualizations")
                    for chart in chart_files:
                        if os.path.exists(chart):
                            st.image(chart, caption=chart.replace('.png', ''))

                with tabs[2]:
                    st.subheader("Detected Anomalies")
                    st.dataframe(anomalies.head(25), use_container_width=True, hide_index=True)

                with tabs[3]:
                    st.subheader("LSTM Predictions")
                    has_ssn = any(col.lower() in ['ssn', 'social security number'] for col in cleaned_df.columns)
                    if not has_ssn:
                        st.warning("No SSN column found; skipping LSTM predictions.")
                    else:
                        try:
                            model_path = 'lstm_model.h5'
                            if os.path.exists(model_path):
                                model = tf.keras.models.load_model(model_path)
                                historical_df = fetch_claims_data()
                                if not historical_df.empty:
                                    all_meds = historical_df['medication'].unique()
                                    med_to_idx = {med: idx + 1 for idx, med in enumerate(all_meds)}
                                    all_ndcs = historical_df['ndc'].dropna().unique()
                                    ndc_to_idx = {ndc: idx + 1 for idx, ndc in enumerate(all_ndcs)}
                                    lstm_predictions = predict_future_meds(cleaned_df, model, med_to_idx, ndc_to_idx)
                                    if lstm_predictions:
                                        prediction_df = pd.DataFrame({
                                            "Hashed SSN": list(lstm_predictions.keys()),
                                            "Kidney Disease Medication Probability (%)": [f"{prob*100:.2f}" for prob in lstm_predictions.values()]
                                        })
                                        st.dataframe(prediction_df, use_container_width=True, hide_index=True)
                                        fig = px.bar(
                                            x=prediction_df["Hashed SSN"],
                                            y=prediction_df["Kidney Disease Medication Probability (%)"],
                                            labels={'x': 'Hashed SSN', 'y': 'Probability (%)'},
                                            title='LSTM Predicted Probability of Kidney Disease Medication',
                                            color_discrete_sequence=['#003087']
                                        )
                                        fig.update_layout(
                                            plot_bgcolor='#D3D3D3',
                                            paper_bgcolor='#FFFFFF',
                                            font_color='#000000',
                                            title_font_color='#003087'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.error("No LSTM predictions available. Ensure sufficient data with hypertension medications.")
                                else:
                                    st.error("No historical claims data found in database.")
                            else:
                                st.error("LSTM model not yet trained. Please train the model first.")
                        except Exception as e:
                            st.error(f"Error loading LSTM model or making predictions: {str(e)}")

                with tabs[4]:
                    st.subheader("Ask About Your Data")
                    user_question = st.text_input("Enter your question")
                    if user_question and client:
                        context = ""
                        if 'numeric_summary' in analysis_results:
                            context += "Numeric Summary:\n" + analysis_results['numeric_summary'].drop(columns=['SSN', 'Social Security Number', 'NDC', 'AWP'], errors='ignore').to_string() + "\n\n"
                        for key, value in analysis_results.items():
                            if key.endswith('_counts') and 'B/G Fill Indicator' not in key:
                                context += f"{key.replace('_counts', '')} Distribution:\n" + value.head(25).to_string() + "\n\n"
                            elif key in ['claims_per_id', 'cost_summary', 'member_medications']:
                                context += f"{key}:\n" + value.head(25).to_string(index=False) + "\n\n"
                        context += "Anomalies:\n" + anomalies.head(25).to_string(index=False) + "\n\n"
                        context += "Predictions:\n" + (prediction_df.to_string(index=False) if 'prediction_df' in locals() else "No LSTM predictions available.")
                        try:
                            response = client.chat.completions.create(
                                model="grok-3",
                                messages=[
                                    {"role": "system", "content": "Answer questions based on pharmacy claims data."},
                                    {"role": "user", "content": f"Context:\n{context}\nQuestion: {user_question}"}
                                ],
                                max_tokens=300
                            )
                            answer = response.choices[0].message.content.strip()
                            st.session_state.chat_history.append((user_question, answer))
                            st.write("**Answer**:")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                    if st.session_state.chat_history:
                        st.write("### History")
                        for q, a in st.session_state.chat_history:
                            st.write(f"**Q**: {q}\n**A**: {a}")

                    if st.button("Clear Chat History"):
                        st.session_state.chat_history = []
                        st.rerun()

            if os.path.exists(temp_file):
                os.remove(temp_file)
else:
    st.error("Please login via website to access dashboard.")
    st.markdown(
        """
        <a href="https://rxaianalytics.com/login" target="_blank" style="color: #1f78b4; text-decoration: underline;">
            Log In Here (Opens in a New Tab)
        </a>
        """,
        unsafe_allow_html=True
    )
    st.write("After logging in, return to this tab and refresh the page to access dashboard.")
