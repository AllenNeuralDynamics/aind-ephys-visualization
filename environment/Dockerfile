FROM continuumio/miniconda3:23.9.0-0

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add packages and reppositories for fonts and libvips (sortingview)
RUN apt-add-repository non-free -y && apt-add-repository contrib -y
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libvips \
        libvips-dev \
        libvips-tools \
        fonts-freefont-ttf=20120503-10 \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.22.1 \
    pyvips==2.2.1 \
    wavpack-numcodecs==0.1.5 \
    spikeinterface[full,widgets]==0.99.1
