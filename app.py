import streamlit as st
import os
from pharmacy_analyzer import main
from openai import OpenAI
import pandas as pd

# Set page title
st.title("Pharmacy Claims Analyzer")

# Initialize xAI client
try:
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
except Exception as e:
    st.error(f"Error initializing AI client: {str(e)}. Please check XAI_API_KEY.")
    client = None

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file:
    # Save uploaded file temporarily
    temp_file = "temp.xlsx"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Run analysis
    cleaned_df, messages, analysis_results, anomalies, chart_files = main(temp_file)
    
    # Display messages
    st.write("### Status")
    st.write(messages)
    
    if cleaned_df is not None:
        # Display analysis results
        st.write("### Claims per Member")
        st.dataframe(analysis_results['claims_per_member'])
        
        st.write("### Brand vs Generic Distribution")
        st.dataframe(analysis_results['brand_generic_dist'].reset_index())
        
        st.write("### Channel Distribution")
        st.dataframe(analysis_results['channel_dist'].reset_index())
        
        st.write("### Average Quantity and Days Supply per Drug")
        st.dataframe(analysis_results['drug_stats'])
        
        # Display anomalies
        st.write("### Detected Anomalies (Possible Errors or Fraud)")
        st.dataframe(anomalies)
        
        # Display charts
        st.write("### Visualizations")
        for chart in chart_files:
            st.image(chart, caption=chart.replace('.png', ''))
        
        # Prepare data context for AI
        context = ""
        context += "Claims per Member:\n" + analysis_results['claims_per_member'].to_string(index=False) + "\n\n"
        context += "Brand vs Generic Distribution:\n" + analysis_results['brand_generic_dist'].to_string() + "\n\n"
        context += "Channel Distribution:\n" + analysis_results['channel_dist'].to_string() + "\n\n"
        context += "Average Quantity and Days Supply per Drug:\n" + analysis_results['drug_stats'].to_string(index=False) + "\n\n"
        context += "Anomalies:\n" + anomalies.to_string(index=False) + "\n\n"
        context += "Raw Data Sample (first 5 rows):\n" + cleaned_df.head().to_string(index=False) + "\n\n"
        
        # Q&A Section
        st.write("### Ask a Question About the Data")
        user_question = st.text_input("Enter your question (e.g., 'How many claims does Member 8 have?' or 'What drugs did Member 2 claim?')")
        
        if user_question:
            if not user_question.strip():
                st.warning("Please enter a valid question.")
            elif client:
                try:
                    # Send question to Grok
                    response = client.chat.completions.create(
                        model="grok-3",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant analyzing pharmacy claims data. Answer questions based on the provided data context. Be concise, accurate, and use the raw data sample for specific claim details."},
                            {"role": "user", "content": f"Data context:\n{context}\n\nQuestion: {user_question}"}
                        ],
                        max_tokens=300
                    )
                    answer = response.choices[0].message.content.strip()
                    # Store in chat history
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
