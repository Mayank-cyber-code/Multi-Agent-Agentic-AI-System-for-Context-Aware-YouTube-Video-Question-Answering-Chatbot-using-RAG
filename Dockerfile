# ===== BASE IMAGE =====
FROM python:3.11-slim

# ===== SET WORKDIR =====
WORKDIR /app

# ===== INSTALL SYSTEM DEPENDENCIES =====
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ===== COPY FILES =====
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# ===== ENV VARIABLES =====
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# ===== RUN APP =====
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]