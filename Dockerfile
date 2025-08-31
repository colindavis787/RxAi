FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Health check for App Runner
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Startup probe to log issues
CMD ["sh", "-c", "echo 'Starting Streamlit' && streamlit run app.py --server.port=8501 --server.headless=true --server.enableCORS=false --server.address=0.0.0.0 2>&1 | tee /tmp/streamlit.log"]
