#!/bin/sh
python -m http.server 80 &
streamlit run BIS_ChatBot.py --server.port=8501 --server.address=0.0.0.0
