FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc supervisor

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY subscriber.py .
COPY utility.py .
COPY chunk_texts.json .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000

CMD ["/usr/bin/supervisord"]
