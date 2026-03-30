# hash:sha256:27dc0782ef58f4c35826cf21da626c8dfdf0f778c81cbd18426a1cbc8115ba18
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV GH_CURATION_REPO=gh://AllenNeuralDynamics/ephys-sorting-manual-curation/main

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        fonts-freefont-ttf=20120503-10build1 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    pyvips==2.2.1 \
    s3fs==2025.5.1 \
    sortingview==0.14.1 \
    spikeinterface[full,widgets]==0.104.0 \
    wavpack-numcodecs==0.2.2
