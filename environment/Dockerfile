# hash:sha256:0a2e37d7d6a981197710b841c6dd29568ecc4e0ab7dd4913b1d44234f5315d20
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

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
    aind-data-schema==0.13.66 \
    pyvips==2.2.1 \
    wavpack-numcodecs==0.1.4

COPY postInstall /
RUN /postInstall
