# hash:sha256:ec6db904f44d1ae607d73705fee6765587de2d14b2099a8b7f2da6b80a10b92b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG KACHERY_CLOUD_CLIENT_ID
ARG KACHERY_CLOUD_PRIVATE_KEY
ARG KACHERY_ZONE

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV GH_CURATION_REPO=gh://AllenNeuralDynamics/ephys-sorting-manual-curation/main

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-freefont-ttf=20120503-10 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.1 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.101.2 \
    wavpack-numcodecs==0.2.1
