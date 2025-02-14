# Dockerfile
FROM python:3.12-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright dependencies and Chromium
RUN playwright install-deps
RUN playwright install chromium

