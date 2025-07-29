FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt
EXPOSE 8200
CMD ["uvicorn", "demo_log:app", "--host", "0.0.0.0", "--port", "8200"]
