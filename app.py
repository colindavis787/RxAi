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

# JWT secret key (will be updated later with the same key used in Flask)
JWT_SECRET_KEY = 'your_jwt_secret_key_12345'  # Replace with a secure key in production

# Load user credentials
credentials_path = Path('.streamlit/credentials.yaml')
try:
    with open(credentials_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    users = config['credentials']['usernames']
except Exception as e:
    st.error(f"Failed to load credentials: {e}")
    users = {}

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
params = st.query_params
username = params.get('username', [None])[0]
token = params.get('token', [None])[0]

# Authenticate user if token is present
if username and token and not st.session_state.authenticated:
    try:
        # Decode and validate the token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        if payload['username'] == username and username in users:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.name = users[username]['name']
        else:
            st.error("Invalid authentication token.")
    except jwt.ExpiredSignatureError:
        st.error("Authentication token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        st.error("Invalid authentication token.")

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
