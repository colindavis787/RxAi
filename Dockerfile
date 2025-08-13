FROM python:3.11-slim
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
