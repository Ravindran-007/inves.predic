# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for Prophet / LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

# Set matplotlib to use Agg backend (no GUI) - fixes font cache issue
ENV MPLBACKEND=Agg
ENV MATPLOTLIBRC=/tmp/matplotlibrc
RUN echo "backend: Agg" > /tmp/matplotlibrc

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Reduce workers to save memory
CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]