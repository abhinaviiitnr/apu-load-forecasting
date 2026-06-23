# Dockerfile — APU Load Forecasting (API + dashboard in one container)
FROM python:3.11-slim

# Avoid Python writing .pyc files / buffering stdout (cleaner logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (only what the app needs at runtime)
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# The API serves both the endpoints and the dashboard on port 8000
EXPOSE 8000

# Run from the api/ directory so imports (forecaster, features) resolve
WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]