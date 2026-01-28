# Use a lightweight python image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose ports for FastAPI (8000) and Gradio (7860)
EXPOSE 8000
EXPOSE 7860

# Command to run both services (FastAPI in background, Gradio in foreground)
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & python app/ui.py