FROM python:3.12-slim AS builder
WORKDIR /build
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/app.py .
RUN mkdir -p /app/models && chown -R appuser /app
USER appuser
ENV MODEL_DIR=/app/models PORT=5000
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]