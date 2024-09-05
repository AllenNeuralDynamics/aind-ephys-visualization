# hash:sha256:fe002c6321c1cdce5aa5048f4e67400af7c0e2e9a63f975996d95bc52821b274
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

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-freefont-ttf=20120503-10 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.101.0 \
    wavpack-numcodecs==0.1.5
