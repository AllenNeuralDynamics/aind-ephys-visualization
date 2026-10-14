# hash:sha256:09765dbeaee918cdeccb92a0237ca0ef2e5504b98f66aebbd4b0fd2b781e1c24
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        fonts-freefont-ttf=20120503-10build1 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    boto3==1.41.5 \
    figpack==0.3.21 \
    figpack-spike-sorting==0.1.15 \
    log-schema==0.2.10.dev1 \
    pyvips==2.2.1 \
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.105.2 \
    wavpack-numcodecs==0.2.2
