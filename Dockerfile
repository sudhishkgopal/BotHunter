#  Stage 1: Install deps into an isolated venv 
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  Stage 2: Minimal runtime image 
FROM python:3.11-slim AS runtime

# Non-root user
RUN groupadd --gid 1000 hunter && \
    useradd  --uid 1000 --gid hunter --create-home hunter

WORKDIR /app

# Only the venv, no pip/build tools/cache
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# App code + config
COPY config.json requirements.txt ./
COPY models.py database.py ingestor.py processor.py cli.py app.py main.py ./
COPY setup.sh ./
RUN chmod +x setup.sh

# Persistent data directory for SQLite
RUN mkdir -p /app/data && chown hunter:hunter /app/data
VOLUME ["/app/data"]

USER hunter

# Default: launch the Streamlit dashboard
EXPOSE 8501
ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
