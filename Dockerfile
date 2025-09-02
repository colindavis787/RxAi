FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc gfortran libblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
ENV PORT=8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
