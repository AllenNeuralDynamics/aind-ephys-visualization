# hash:sha256:5e6d592beef8b5d68a2806e310e30632c0d73ece645e08f37d69fba34c46c3de
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
    aind-data-schema==0.13.50 \
    pyvips==2.2.1 \
    wavpack-numcodecs==0.1.4

COPY postInstall /
RUN /postInstall
