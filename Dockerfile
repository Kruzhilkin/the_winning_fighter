FROM python:3.10-slim

COPY . ./root
COPY ./requirements.txt /root/requirements.txt
WORKDIR /root

# обновим pip и установим библиотеки из requirements, --ignore-installed - переустановка пакетов, если они уже есть
RUN pip install --upgrade pip && \
    pip install --ignore-installed -r /root/requirements.txt
#RUN pip install flask gunicorn numpy sklearn scipy pandas requests flask_wtf