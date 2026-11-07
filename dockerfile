FROM python:3.11.15-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=10

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src/requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src

WORKDIR /app/src

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]