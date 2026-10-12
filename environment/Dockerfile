<<<<<<< HEAD
# hash:sha256:6ca06d73b992b8c146a49a3aa12af7518547e66fee3bac1d568ee2773f4a694c
=======
# hash:sha256:308261e0db5a66ad3c2f9c0605736f84a3bd51d36d680dae8adedf5d7d0ea997
>>>>>>> main
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV FIGPACK_UPLOAD=1
ENV FIGPACK_API_KEY=35725b3d95eacc9c1c00895ba2ef051110d26925cf0b61cba3060806faa81783
ENV FIGPACK_BUCKET=aind-figpack-client

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        fonts-freefont-ttf=20120503-10build1 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
<<<<<<< HEAD
    aind-log-utils==0.2.3 \
    figpack==0.3.20 \
    figpack-spike-sorting==0.1.15 \
    pyvips==2.2.1 \
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.104.8 \
=======
    log-schema==0.2.10.dev1 \
    pyvips==2.2.1 \
    s3fs==2025.5.1 \
    sortingview==0.14.2 \
    spikeinterface[full,widgets]==0.105.0 \
>>>>>>> main
    wavpack-numcodecs==0.2.2
