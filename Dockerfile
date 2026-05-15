FROM python:3.11-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

ENV _JAVA_OPTIONS="-XX:UseSVE=0 -Xshare:off -XX:+UseContainerSupport -XX:MaxRAMPercentage=50"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk \
    build-essential \
    gcc \
    g++ \
    graphviz \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -d "/usr/lib/jvm/java-21-openjdk-arm64" ]; then \
        ln -sf /usr/lib/jvm/java-21-openjdk-arm64 /usr/lib/jvm/default-java; \
    elif [ -d "/usr/lib/jvm/java-21-openjdk-amd64" ]; then \
        ln -sf /usr/lib/jvm/java-21-openjdk-amd64 /usr/lib/jvm/default-java; \
    fi

ENV JAVA_HOME=/usr/lib/jvm/default-java

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]