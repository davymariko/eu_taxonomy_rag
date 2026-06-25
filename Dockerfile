FROM python:3.11-slim AS base

WORKDIR /app

# Needed by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/   data/
COPY src/    src/

# Helps the vector store across runs via a volume
VOLUME ["/app/vectorstore"]

# run CLI
CMD ["python", "src/app.py"]
