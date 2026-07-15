# olmoearth-pretrain requires torch>=2.7,<2.8 — keep the base torch inside that
# range so pip installs don't replace the baked-in torch.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS base

ARG GELOS_VERSION=v0.3.7

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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

WORKDIR /app
ENV PYTHONPATH=/app

RUN uv pip install --system --no-cache awscli boto3 mkdocs ruff pytest
# olmoearth-pretrain is explicit until GELOS_VERSION points at a gelos release
# that declares it as a core dependency.
RUN uv pip install --system --no-cache \
    "gelos[alphaearth] @ git+https://github.com/ClarkCGA/gelos.git@${GELOS_VERSION}" \
    "olmoearth-pretrain~=0.1.1"

COPY pyproject.toml README.md Makefile LICENSE /app/
COPY src/ /app/src/
RUN uv pip install --system --no-cache --no-deps -e . && \
    chmod -R a+w /app

FROM base AS test

COPY tests/ /app/tests/
RUN chmod -R a+w /app/tests

CMD ["python", "-m", "pytest", "tests"]

FROM base AS prod

CMD ["make", "-h"]

FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11 AS dev

ARG GELOS_VERSION=v0.3.7

USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    make \
    git \
    build-essential \
    libsqlite3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/felt/tippecanoe.git /tmp/tippecanoe && \
    make -C /tmp/tippecanoe -j && \
    make -C /tmp/tippecanoe install && \
    rm -rf /tmp/tippecanoe

WORKDIR /app

RUN uv pip install --system --no-cache awscli boto3 mkdocs ruff pytest
RUN uv pip install --system --no-cache \
    "gelos[alphaearth] @ git+https://github.com/ClarkCGA/gelos.git@${GELOS_VERSION}"

# The jupyter base ships torch 2.5/cu121, but olmoearth-pretrain requires
# torch>=2.7,<2.8. Bake the matched torch/torchvision pair into the image so
# the container-start gelos install resolves without touching torch.
RUN uv pip install --system --no-cache \
    torch==2.7.1 torchvision==0.22.1 "olmoearth-pretrain~=0.1.1"

COPY pyproject.toml README.md Makefile LICENSE /app/
COPY src/ /app/src/
RUN uv pip install --system --no-cache --no-deps -e .

CMD ["start-notebook.py"]
