# hash:sha256:ba83b65c116fd8a9316e2359c52c4575ee5bae26f5c102dba553a55b045b136d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

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
    aind-data-schema==0.22.1 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.99.1 \
    wavpack-numcodecs==0.1.5
