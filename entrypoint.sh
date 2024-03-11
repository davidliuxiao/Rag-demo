#!/bin/sh
python -m http.server 80 &
streamlit run BIS_ChatBot.py --server.port=8501
