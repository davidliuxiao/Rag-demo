FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/davidliuxiao/Rag-demo.git .

RUN pip3 install -r requirements.txt

#COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh


EXPOSE 80
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["./entrypoint.sh"]
#ENTRYPOINT ["streamlit", "run", "BIS_ChatBot.py", "--server.port=8501", "--server.address=0.0.0.0"]

#CMD python -m http.server 80 &