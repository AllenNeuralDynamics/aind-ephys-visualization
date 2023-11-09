# hash:sha256:1674e76e80cf466a83a1985989a571bae03dc7d7d01a2b9aaa8d1fee9b5b401f
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
    aind-data-schema==0.15.25 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.99.0 \
    wavpack-numcodecs==0.1.5
