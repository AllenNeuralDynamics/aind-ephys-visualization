FROM continuumio/miniconda3:23.9.0-0

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-freefont-ttf=20120503-10 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.15.25 \
    pyvips==2.2.1 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
