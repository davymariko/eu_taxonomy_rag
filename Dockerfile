FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies (needed by faiss-cpu)
RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    libgomp1

RUN rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY data/ data/
COPY src/ src/

# Keeps vector store
VOLUME ["/app/vectorstore"]

# Run CLI
CMD ["python", "src/app.py"]