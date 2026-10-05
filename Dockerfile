# single gelos pin shared by every stage — re-declared bare inside each stage
# that uses it. v0.9.0 adds the kNN geographic-distance metric and plots
# (gelos#85: `knn_geo_distance`, `knn_gsd_plot`, `knn_lat_diff_plot`); v0.8.0 is required for the Prithvi TL wrapper
# (gelos#78 / PR #90: `prithvi_eo_v2_*_tl_coords`, consumed by exp038/exp039
# via GELOSLCTimeLocationDataSet) and also brings per-config figure folders
# (gelos#89 / PR #91: figures land in `figures/{data_version}/{config_stem}/`
# and `figures/comparisons/{config_stem}/`). On top of v0.7.0's DINOv3 backbones
# and GELOSDataSet `clip_range_bands` (gelos#16/#88, forwarded unconditionally
# by GELOSLCDataSet) and v0.6.0's dataset timestamp/location hooks (gelos#79,
# used by GELOSLCMetadataDataSet) and OlmoEarth patch nodata masking (gelos#86).
# Keep in sync with the gelos pin in pyproject.toml's
# [tool.pixi.pypi-dependencies].
ARG GELOS_VERSION=v0.11.0

# olmoearth-pretrain (a gelos core dependency) requires torch>=2.7,<2.8 — keep
# the base torch inside that range so pip installs don't replace the baked-in
# torch.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS base

ARG GELOS_VERSION

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

# --upgrade so a boto3 baked into the base image can't stay behind the exact
# botocore that awscli pins, which breaks boto3's import
RUN uv pip install --system --no-cache --upgrade awscli boto3 mkdocs ruff pytest
RUN uv pip install --system --no-cache \
    "gelos[alphaearth] @ git+https://github.com/ClarkCGA/gelos.git@${GELOS_VERSION}"

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

ARG GELOS_VERSION

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

# --upgrade so a boto3 baked into the base image can't stay behind the exact
# botocore that awscli pins, which breaks boto3's import
RUN uv pip install --system --no-cache --upgrade awscli boto3 mkdocs ruff pytest

# The jupyter base ships torch 2.5/cu121, but olmoearth-pretrain (a gelos core
# dependency) requires torch>=2.7,<2.8. Bake the matched torch/torchvision
# pair first so the gelos installs (here and at container start) resolve
# against it without touching torch.
RUN uv pip install --system --no-cache torch==2.7.1 torchvision==0.22.1
RUN uv pip install --system --no-cache \
    "gelos[alphaearth] @ git+https://github.com/ClarkCGA/gelos.git@${GELOS_VERSION}"

COPY pyproject.toml README.md Makefile LICENSE /app/
COPY src/ /app/src/
RUN uv pip install --system --no-cache --no-deps -e .

CMD ["start-notebook.py"]
