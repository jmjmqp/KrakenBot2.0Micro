FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
EXPOSE 5000

CMD gunicorn --worker-class gthread --threads 2 --bind 0.0.0.0:$PORT app:app
