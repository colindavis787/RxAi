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
    cleaned_df, messages, analysis_results, anomalies, chart_files, predictions = main(temp_file)
    
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
                    st.write(f"**Member {member} Predicted Cost (with 5% annual inflation):**")
                    st.write(f"Month 1: ${value[0]:.2f}, Month 2: ${value[1]:.2f}, Month 3: ${value[2]:.2f}")
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
        user_question = st.text_input("Enter your question (e.g., 'What’s the predicted cost for Member 9?' or 'How many claims per ID?')")
        
        if user_question:
            if not user_question.strip():
                st.warning("Please enter a valid question.")
            elif client:
                try:
                    response = client.chat.completions.create(
                        model="grok-3",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant analyzing pharmacy claims data. Answer questions based on the provided data context, covering any columns and predictions. Be concise and accurate."},
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