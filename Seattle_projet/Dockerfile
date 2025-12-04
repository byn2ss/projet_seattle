FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train_and_save.py

EXPOSE 3000

CMD ["bentoml", "serve", "api_service:svc", "--host", "0.0.0.0", "--port", "3000"]
