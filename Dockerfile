FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    build-essential \
    gcc \
    g++ \
    graphviz \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]