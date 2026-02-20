FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base

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

COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt
RUN uv pip install --system --no-cache "git+https://github.com/ClarkCGA/gelos.git"

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

COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

COPY pyproject.toml README.md Makefile LICENSE /app/
COPY src/ /app/src/
RUN uv pip install --system --no-cache --no-deps -e .

CMD ["start-notebook.py"]
