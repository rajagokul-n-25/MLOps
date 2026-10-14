FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir mlflow psycopg2-binary "psycopg[binary]"

WORKDIR /mlflow

RUN mkdir -p /mlflow/data/artifacts

EXPOSE 5000
