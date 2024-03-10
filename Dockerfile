FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/davidliuxiao/Rag-demo.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "BIS_ChatBot.py", "--server.port=8501", "--server.address=0.0.0.0"]

# use Nginx to serve static files
FROM nginx:alpine
RUN mkdir -p /usr/share/nginx/html/pdf
COPY --from=builder /app/static/data/pdfs /usr/share/nginx/html/pdfs
EXPOSE 80