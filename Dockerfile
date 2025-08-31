FROM python:3.11-slim
<<<<<<< HEAD

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Health check for App Runner
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Startup probe to log issues
CMD ["sh", "-c", "echo 'Starting Streamlit' && streamlit run app.py --server.port=8501 --server.headless=true --server.enableCORS=false --server.address=0.0.0.0 2>&1 | tee /tmp/streamlit.log"]
=======
WORKDIR /app
RUN apt-get update && apt-get install -y gcc gfortran libblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
ENV PORT=8501
# Disable development mode
ENV FLASK_ENV=production
CMD gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 60 app:app
>>>>>>> 41c87d72dd7eb52f51936fb2df2f7a4a7b87eb66
