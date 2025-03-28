# hash:sha256:c11e1039d6c7a8461a3d4cf4e6b7f3894daf86f7159383b3c1643cb2ea83d1d1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG KACHERY_ZONE

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-freefont-ttf=20120503-10 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.0 \
    numpy==1.26.4 \
    pyvips==2.2.1 \
    sortingview==0.14.1 \
    spikeinterface[full,widgets]==0.100.7 \
    wavpack-numcodecs==0.1.5
