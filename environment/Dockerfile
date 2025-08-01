# hash:sha256:4f8956eaf3d1296ed1fd5bedec347f117cb65f3929009ab0b470c953f08abf9c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG KACHERY_API_KEY
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
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    pyvips==2.2.1 \
    s3fs==2025.5.1 \
    sortingview==0.14.1 \
    spikeinterface[full,widgets]==0.102.3 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
