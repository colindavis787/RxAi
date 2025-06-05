import streamlit as st
import os
from pharmacy_analyzer import main
from openai import OpenAI
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import jwt
import datetime
from pathlib import Path
import logging
from urllib.parse import unquote
import psycopg

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

# Initialize session state for authentication and chat history
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
    # Display welcome message and logout link
    st.write(f'Welcome *{st.session_state["name"]}*!')
    st.sidebar.markdown("[Logout](https://rxaianalytics.com/logout)")

    # Initialize xAI client
    try:
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    except Exception as e:
        st.error(f"Error initializing AI client: {str(e)}. Please check XAI_API_KEY.")
        client = None

    # Set page title
    st.title("Pharmacy Claims Analyzer")

    # Display historical uploads
    st.write("### Past Uploads")
    try:
        conn = sqlite3.connect('claims_history.db')
        past_uploads = pd.read_sql("SELECT DISTINCT upload_id, upload_date FROM claims", conn)
        conn.close()
        if not past_uploads.empty:
            st.dataframe(past_uploads)
        else:
            st.write("No past uploads found.")
    except Exception as e:
        st.error(f"Error accessing historical data: {str(e)}")

    # Inflation rate slider
    inflation_rate = st.slider("Annual Drug Price Inflation (%)", 0.0, 10.0, 5.0) / 100

    # File uploader
    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

    if uploaded_file:
        # Validate file size (limit to 10MB)
        if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
            st.error("File size exceeds 10MB limit.")
        else:
            # Save uploaded file temporarily
            temp_file = "temp.xlsx"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run analysis with inflation rate
            cleaned_df, messages, analysis_results, anomalies, chart_files, predictions = main(temp_file, inflation_rate)

            # Display messages
            st.write("### Status")
            st.write(messages)

            if cleaned_df is not None:
                # Display analysis results
                if 'numeric_summary' in analysis_results:
                    st.write("### Numeric Column Summary")
                    st.dataframe(analysis_results['numeric_summary'])

                for key, value in analysis_results.items():
                    if key.endswith('_counts'):
                        st.write(f"### {key.replace('_counts', '')} Distribution")
                        st.dataframe(value.reset_index(name='Count'))

                    elif key == 'claims_per_id':
                        st.write("### Claims per ID")
                        st.dataframe(value)

                    elif key == 'cost_summary':
                        st.write("### Total Cost per ID")
                        st.dataframe(value)

                    elif key == 'member_medications':
                        st.write("### Medications and Likely Conditions per Member")
                        st.dataframe(value)

                # Display anomalies
                st.write("### Detected Anomalies")
                st.dataframe(anomalies)

                # Display predictions
                st.write("### Future Utilization and Cost Predictions (Next 3 Months)")
                if predictions:
                    for key, value in predictions.items():
                        if key.endswith('_utilization'):
                            member = key.replace('_utilization', '')
                            st.write(f"**Member {member} Predicted Utilization (Claims):**")
                            st.write(f"Month 1: {value[0]:.2f}, Month 2: {value[1]:.2f}, Month 3: {value[2]:.2f}")
                        elif key.endswith('_cost'):
                            member = key.replace('_cost', '')
                            st.write(f"**Member {member} Predicted Cost (with {inflation_rate*100}% annual inflation):**")
                            st.write(f"Month 1: ${value[0]:.2f}, Month 2: ${value[1]:.2f}, Month 3: ${value[2]:.2f}")

                    # Plot predictions
                    plt.figure(figsize=(10, 6))
                    for key, value in predictions.items():
                        if key.endswith('_utilization'):
                            member = key.replace('_utilization', '')
                            plt.plot([1, 2, 3], value, label=f"Member {member} Utilization")
                        elif key.endswith('_cost'):
                            member = key.replace('_cost', '')
                            plt.plot([1, 2, 3], value, '--', label=f"Member {member} Cost ($)")
                    plt.title('Predicted Utilization and Cost (Next 3 Months)')
                    plt.xlabel('Month')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.savefig('prediction_plot.png')
                    plt.close()
                    st.image('prediction_plot.png', caption='Prediction Plot')
                else:
                    st.write("No predictions available. Ensure ID, date, and quantity columns exist.")

                # Display charts
                st.write("### Visualizations")
                for chart in chart_files:
                    st.image(chart, caption=chart.replace('.png', ''))

                # Prepare data context for AI
                context = ""
                if 'numeric_summary' in analysis_results:
                    context += "Numeric Column Summary:\n" + analysis_results['numeric_summary'].to_string() + "\n\n"
                for key, value in analysis_results.items():
                    if key.endswith('_counts'):
                        context += f"{key.replace('_counts', '')} Distribution:\n" + value.to_string() + "\n\n"
                    elif key == 'claims_per_id':
                        context += "Claims per ID:\n" + value.to_string(index=False) + "\n\n"
                    elif key == 'cost_summary':
                        context += "Total Cost per ID:\n" + value.to_string(index=False) + "\n\n"
                    elif key == 'member_medications':
                        context += "Medications and Conditions per Member:\n" + value.to_string(index=False) + "\n\n"
                context += "Raw Data Sample (first 5 rows):\n" + cleaned_df.head().to_string(index=False) + "\n\n"
                context += "Anomalies:\n" + anomalies.to_string(index=False) + "\n\n"
                context += "Predictions (Next 3 Months):\n"
                for key, value in predictions.items():
                    if key.endswith('_utilization'):
                        member = key.replace('_utilization', '')
                        context += f"Member {member} Utilization: Month 1: {value[0]:.2f}, Month 2: {value[1]:.2f}, Month 3: {value[2]:.2f}\n"
                    elif key.endswith('_cost'):
                        member = key.replace('_cost', '')
                        context += f"Member {member} Cost: Month 1: ${value[0]:.2f}, Month 2: ${value[1]:.2f}, Month 3: ${value[2]:.2f}\n"

                # Q&A Section
                st.write("### Ask a Question About the Data")
                user_question = st.text_input("Enter your question (e.g., 'What condition is Member 9’s medication treating?' or 'How many claims per ID?')")

                if user_question:
                    if not user_question.strip():
                        st.warning("Please enter a valid question.")
                    elif client:
                        try:
                            response = client.chat.completions.create(
                                model="grok-3",
                                messages=[
                                    {"role": "system", "content": "You are an AI assistant analyzing pharmacy claims data. Answer questions based on the provided data context, covering any columns, predictions, and medication conditions. Be concise and accurate."},
                                    {"role": "user", "content": f"Data context:\n{context}\n\nQuestion: {user_question}"}
                                ],
                                max_tokens=300
                            )
                            answer = response.choices[0].message.content.strip()
                            st.session_state.chat_history.append((user_question, answer))
                            st.write("**Answer**")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error getting AI response: {str(e)}")
                    else:
                        st.error("AI client not initialized. Check XAI_API_KEY.")

                # Display chat history
                if st.session_state.chat_history:
                    st.write("### Chat History")
                    for q, a in st.session_state.chat_history:
                        st.write(f"**Q**: {q}\n**A**: {a}")

                # Clear chat history button
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
else:
    st.error("Please log in via the website to access the dashboard.")
    st.markdown(
        """
        <a href="https://rxaianalytics.com/login" target="_blank" style="color: #1f77b4; text-decoration: underline;">
            Log In Here (Opens in a New Tab)
        </a>
        """,
        unsafe_allow_html=True
    )
    st.write("After logging in, return to this tab and refresh the page to access the dashboard.")
