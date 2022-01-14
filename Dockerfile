FROM python:3.10-slim

COPY . ./root

WORKDIR /root

RUN pip install flask gunicorn numpy sklearn scipy pandas requests