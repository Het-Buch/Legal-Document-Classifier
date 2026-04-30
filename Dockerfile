# Dockerfile (CPU)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy repo
COPY . /app

# Install python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose ports for streamlit and fastapi
EXPOSE 8501 8000

# Default entrypoint runs the API
CMD ["bash", "-lc", "uvicorn deployment.fastapi_server:app --host 0.0.0.0 --port 8000"]
