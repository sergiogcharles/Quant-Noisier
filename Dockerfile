FROM python:3.8-slim-buster
WORKDIR ~/Quant-Noisier
COPY . .
RUN apt-get update && apt-get install -y gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++
RUN pip install --editable ./
ENTRYPOINT sh roberta-wiki-quant-noise.sh
