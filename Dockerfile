# Install dependencies in venv and copy to final image
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Start final image
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 hunter && \
    useradd --uid 1000 --gid hunter --create-home hunter

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY models.py database.py processor.py cli.py main.py ./

VOLUME ["/app/data"]
ENV DATABASE_URL="sqlite:///data/bothunter.db"

USER hunter

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]