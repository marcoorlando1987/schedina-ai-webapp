# Immagine unica: serve sia l'app web (default) sia il job di aggiornamento.
#   app:  docker run -p 8080:8080 -v $PWD/data:/data IMAGE
#   job:  docker run -e FOOTBALL_DATA_API_TOKEN=... -v $PWD/data:/data IMAGE python -m schedina_ai run
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SCHEDINA_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# xgboost-cpu: stesso modulo `xgboost`, senza le librerie CUDA (immagine molto più leggera)
RUN sed -i 's/^xgboost.*/xgboost-cpu>=2.1/' requirements.txt && pip install -r requirements.txt

COPY schedina_ai ./schedina_ai
COPY app.py ./
COPY .streamlit ./.streamlit

RUN useradd -m app && mkdir -p /data && chown app /data
USER app

EXPOSE 8080
# Cloud Run imposta $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
