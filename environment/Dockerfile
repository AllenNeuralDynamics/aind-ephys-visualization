# hash:sha256:9a27fd2745c773f9e90aec74edaa5c1d7fa7bf0dbce9908a78ab7b18dbfbc233
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG KACHERY_CLOUD_CLIENT_ID
ARG KACHERY_CLOUD_PRIVATE_KEY
ARG KACHERY_ZONE

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.100.0 \
    wavpack-numcodecs==0.1.5

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.22.1 \
    pyvips==2.2.1 \
    wavpack-numcodecs==0.1.5 \
    spikeinterface[full,widgets]==0.99.1
