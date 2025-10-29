FROM python:3.13-slim

WORKDIR /app

COPY . /app

# Install git and other essentials before pip install
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["python3", "flask_api/app.py"]