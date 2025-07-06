import streamlit as st
import os
from pharmacy_analyzer import main
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# JWT secret key
JWT_SECRET_KEY = 'your_jwt_secret_key_12345'

# Load user credentials from database
def load_users():
    try:
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        conninfo = {
            'dbname': url.split('/')[3],
            'user': url.split('//')[1].split(':')[0],
            'password': url.split('//')[1].split(':')[1].split('@')[0],
            'host': url.split('@')[1].split(':')[0],
            'port': url.split(':')[3].split('/')[0],
            'sslmode': 'require'
        }
        logger.debug("Connecting to Postgres database")
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
        st.error(f"Failed to load users: {str(e)}")
        return {}

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
logger.debug(f"Extracted token before decoding: {token}")
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
                st.session_state.name = users[username]['name']
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
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main { background-color: #FFFFFF; padding: 20px; border-radius: 10px; }
        .stButton>button { background-color: #003087; color: white; border-radius: 5px; }
        .stDataFrame { max-height: 400px; overflow-y: auto; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Pharmacy Claims Analyzer")
    st.markdown(f"Welcome, *{st.session_state['name']}*!", unsafe_html=True)
    st.sidebar.markdown("[Logout](https://rxaianalytics.com/logout)")

    # Initialize xAI client
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
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            cleaned_df, messages, analysis_results, anomalies, chart_files, predictions = main(temp_file, inflation_rate)
            st.subheader("Processing Status")
            st.write(messages)

            if cleaned_df is not None:
                tabs = st.tabs(["Summary", "Visualizations", "Anomalies", "Predictions", "Q&A"])
                with tabs[0]:
                    st.subheader("Analysis Summary")
                    if 'numeric_summary' in analysis_results:
                        st.write("Numeric Column Summary")
                        # Remove specified columns
                        numeric_summary = analysis_results['numeric_summary'].drop(columns=['Member Number', 'NDC', 'AWP'], errors='ignore')
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
                    st.subheader(f"Predicted Costs at {inflation_rate*100:.1f}% Inflation")
                    if predictions:
                        current_year = min(int(k) for k in predictions.keys())
                        year_labels = {
                            str(current_year): f"Current Year ({current_year})",
                            str(current_year + 1): f"Year 1 ({current_year + 1})",
                            str(current_year + 2): f"Year 2 ({current_year + 2})",
                            str(current_year + 3): f"Year 3 ({current_year + 3})"
                        }
                        prediction_df = pd.DataFrame({
                            "Year": [year_labels.get(k, k) for k in predictions.keys()],
                            "Predicted Cost ($)": [f"${cost:,.2f}" for cost in predictions.values()]
                        })
                        st.table(prediction_df.style.set_properties(**{
                            'background-color': '#003087',
                            'color': 'white',
                            'text-align': 'center'
                        }))
                        fig = px.line(
                            x=[year_labels.get(k, k) for k in predictions.keys()],
                            y=[cost for cost in predictions.values()],
                            labels={'x': 'Year', 'y': 'Predicted Cost ($)'},
                            title=f'Cost Trend Forecast at {inflation_rate*100:.1f}% Inflation',
                            markers=True
                        )
                        fig.update_traces(line_color='#003087', marker=dict(size=10, color='#4B5EAA'))
                        fig.update_layout(
                            plot_bgcolor='#D3D3D3',
                            paper_bgcolor='#FFFFFF',
                            font_color='#000000',
                            title_font_color='#003087'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No predictions available. Check processing summary for errors.")

                with tabs[4]:
                    st.subheader("Ask About Your Data")
                    user_question = st.text_input("Enter your question")
                    if user_question and client:
                        context = ""
                        if 'numeric_summary' in analysis_results:
                            context += "Numeric Summary:\n" + analysis_results['numeric_summary'].drop(columns=['Member Number', 'NDC', 'AWP'], errors='ignore').to_string() + "\n\n"
                        for key, value in analysis_results.items():
                            if key.endswith('_counts') and 'B/G Fill Indicator' not in key:
                                context += f"{key.replace('_counts', '')} Distribution:\n" + value.head(25).to_string() + "\n\n"
                            elif key in ['claims_per_id', 'cost_summary', 'member_medications']:
                                context += f"{key}:\n" + value.head(25).to_string(index=False) + "\n\n"
                        context += "Anomalies:\n" + anomalies.head(25).to_string(index=False) + "\n\n"
                        context += "Predictions:\n" + prediction_df.to_string(index=False) + "\n"
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
