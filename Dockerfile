FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    make \
    curl \
    git \
    build-essential \
    libsqlite3-dev \
    zlib1g-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
    
RUN git clone https://github.com/felt/tippecanoe.git /tmp/tippecanoe && \
    make -C /tmp/tippecanoe -j && \
    make -C /tmp/tippecanoe install && \
    rm -rf /tmp/tippecanoe

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local sh

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml README.md Makefile LICENSE /app/
COPY src/ /app/src/
COPY gelos/ /app/gelos/
RUN CONDA_OVERRIDE_CUDA="12.8" pixi install
RUN chmod -R a+rwX /app/.pixi

FROM base AS test

COPY tests/ /app/tests/

CMD ["pixi", "run", "make", "test"]

FROM base AS prod

CMD ["pixi", "run", "make", "-h"]

FROM quay.io/jupyter/pytorch-notebook:python-3.11 AS dev

USER root

RUN apt-get update \
    && apt-get install -y \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local sh
ENV PATH="/app/.pixi/envs/default/bin:${PATH}"
ENV CONDA_OVERRIDE_CUDA="12.8"
WORKDIR /app

CMD ["pixi", "run", "start-notebook.py"]