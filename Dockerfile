FROM python:3.13-slim

WORKDIR /app

COPY . /app

# Install git and other essentials before pip install
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]