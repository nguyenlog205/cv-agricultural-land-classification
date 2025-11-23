FROM continuumio/miniconda3:latest

WORKDIR /app
COPY . /app/

RUN conda env create -f /app/environment.yml -n capstone-geospatial-ai \
    && conda clean -a
