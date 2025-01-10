# hash:sha256:866ad697c0a92241d7e9c5d97ce1dd1e8c5ac574c6d7d75c5be42cc146652863
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
    aind-log-utils==0.2.3 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.101.2 \
    wavpack-numcodecs==0.2.1
