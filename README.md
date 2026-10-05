# GELOS-LC

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Geospatial Exploration of Latent Observation Space for Land Cover (GELOS-LC): evaluates
geospatial foundation models (Prithvi EO V2, TerraMind V1, OlmoEarth, AlphaEarth, DINOv3) and
spectral-band baselines on a multi-sensor (Sentinel-2, Sentinel-1, Landsat, DEM) land cover
dataset. The pipeline is driven by the [`gelos`](https://github.com/ClarkCGA/gelos) library;
this repo holds the dataset class, experiment configs, and results.

## Setup

Create a `.env` in the repo root (read by both `docker compose` and the `Makefile`):

```
RAW_PATH=/path/to/data/raw
PROCESSED_PATH=/path/to/data/processed
INTERIM_PATH=/path/to/data/interim
EXTERNAL_PATH=/path/to/data/external
GELOS_MODULE_PATH=/path/to/gelos        # local gelos checkout, mounted into the dev service
JUPYTER_HOST_PORT=8888
DASK_HOST_PORT=8787
JUPYTERLAB_ARGS=
UID=1000
GID=100
GELOS_BUCKET=...                        # S3 upload of app files
AWS_ACCESS_KEY=...
AWS_SECRET_KEY=...
AWS_REGION=...
```

`HF_TOKEN`, `DINOV3_VITB16_CKPT` and `DINOV3_VITL16_SAT_CKPT` are also passed through for the
DINOv3 experiments (checkpoint paths must be in-container paths, e.g. under
`/app/data/external/`).

Build the images (rebuild after bumping `GELOS_VERSION` in the `Dockerfile`):

```bash
docker compose build
```

There are three compose services:

- `prod` — gelos baked in at the pinned `GELOS_VERSION`; used for pipeline runs.
- `dev` — Jupyter Lab image that reinstalls the local gelos checkout (`GELOS_MODULE_PATH`)
  at container start; use it when testing unreleased gelos changes.
- `test` — runs `pytest tests`.

For a local (non-Docker) environment, use [pixi](https://pixi.sh):

```bash
pixi install              # or: make requirements
make dev-install          # optional: overlay an editable ../gelos checkout
```

## Running the pipeline

Each experiment config (`configs/expNNN_*.yaml`) goes through **generation** (embeddings
written to `INTERIM_PATH`) and **analysis** (transforms, kNN / downstream models and figures
written to `PROCESSED_PATH` and `reports/figures/`). **Comparison** configs
(`configs/comparisons/*.yaml`) then compare results across experiments.

### Full pipeline across GPUs

`scripts/run_pipeline.sh` runs generation + analysis for each config, spread round-robin
across GPUs, then runs all comparisons once every config has succeeded. Logs go to
`generation_logs/`. Already-complete steps are skipped unless `OVERWRITE` is set.

```bash
scripts/run_pipeline.sh                    # every config in configs/
scripts/run_pipeline.sh 30 31              # by experiment number (exp030_*, exp031_*)
PER_GPU=4 scripts/run_pipeline.sh          # 4 concurrent configs per GPU
SERVICE=dev scripts/run_pipeline.sh 34 35  # use the dev image (local gelos checkout)
OVERWRITE=1 scripts/run_pipeline.sh 34 35  # recompute even if complete
GPUS=0,1 scripts/run_pipeline.sh           # restrict to specific GPUs
```

`SERVICE` defaults to `prod`.

### Individual steps

```bash
# one experiment
docker compose run --rm prod python -m gelos.generation -y configs/exp024_olmoearth_v1_base_s2.yaml
docker compose run --rm prod python -m gelos.analysis   -y configs/exp024_olmoearth_v1_base_s2.yaml

# analysis for every config in configs/
docker compose run --rm prod make analysis

# all comparisons, or a single one
docker compose run --rm prod python -m gelos.comparison
docker compose run --rm prod python -m gelos.comparison -y configs/comparisons/03_prithvi_tl_knn.yaml
```

Without `-y`, generation and analysis process every YAML in `/app/configs` and comparison
processes every YAML in `/app/configs/comparisons`. Pass `--overwrite` to generation/analysis
to recompute.

Downstream random-forest accuracy comparisons for specific experiment sets:

```bash
docker compose run --rm prod make compare-model-results             # OlmoEarth v1.2 generic vs real dates
docker compose run --rm prod make compare-model-results-prithvi-tl  # Prithvi vs Prithvi TL
```

### GELOS web app files

Generate the app's JSON, PMTiles and `config.js`, then upload them to `s3://gelos-fm/`:

```bash
pixi run make generate-app-files
pixi run make upload-app-files
```

These also work inside Docker (`docker compose run --rm prod make generate-app-files`).

### Interactive development

```bash
docker compose up dev                     # Jupyter Lab on JUPYTER_HOST_PORT
docker compose run --rm -it dev bash      # shell with the local gelos checkout installed
```

### Tests and linting

```bash
docker compose run --rm test   # or locally: make test
make lint                      # ruff format --check && ruff check
make format                    # ruff check --fix && ruff format
```

Run `make` with no arguments to list all targets.

## Project Organization

```
├── CLAUDE.md          <- Guidance for Claude Code when working in this repo
├── Dockerfile         <- base / test / prod / dev image stages; pins GELOS_VERSION
├── LICENSE
├── Makefile           <- Convenience targets (analysis, comparison, app files, lint, test)
├── README.md
├── compose.yml        <- dev, test and prod services with data-path mounts and GPU access
├── pixi.lock
├── pyproject.toml     <- Package metadata, ruff config, and pixi environment (pins gelos)
│
├── configs
│   ├── expNNN_*.yaml  <- One config per experiment (exp001–exp039): data module, model,
│   │                     and embedding extraction strategies with transforms, plots, models
│   └── comparisons    <- Cross-experiment comparison configs (01a_* … 06d_*), including
│                         *_spatial_knn variants
│
├── custom_modules     <- Placeholder package for custom modules
│
├── data               <- Mount points for RAW_PATH, INTERIM_PATH, PROCESSED_PATH,
│   ├── external          EXTERNAL_PATH
│   ├── interim        <- Generated embeddings
│   ├── processed      <- Analysis outputs (transformed embeddings, metrics, app files)
│   └── raw            <- GELOS-LC chips (.tif) and gelos_chip_tracker.geojson
│
├── docker
│   └── 10-install-gelos.sh   <- dev-container start hook: editable-installs /app/gelos
│
├── docs               <- mkdocs project
│
├── models
│   └── prithvi_eo_v2.py      <- Prithvi EO V2 model code (IBM, Apache 2.0)
│
├── notebooks          <- Walkthroughs of individual pipeline steps
│   ├── 00_dwg_exploredataset.ipynb
│   ├── 01_dwg_calculatestatistics.ipynb
│   ├── 02_dwg_generateembeddings.ipynb
│   ├── 03_dwg_analyzeembeddings.ipynb
│   ├── 04_dwg_generateappfiles.ipynb
│   ├── 05_dwg_interactiveplotting.ipynb
│   └── 06_dwg_alphaearth_walkthrough.ipynb
│
├── references
│
├── reports
│   └── figures
│       ├── comparisons       <- One folder per comparison config
│       └── v0.50.1           <- One folder per experiment config (data version v0.50.1)
│
├── scripts
│   └── run_pipeline.sh       <- Multi-GPU generation + analysis, then comparisons
│
├── src                <- Installable package (`src`)
│   ├── __init__.py
│   ├── app_files_generation.py   <- Generate GELOS app files (JSON, PMTiles, config.js)
│   ├── app_files_upload.py       <- Upload GELOS app files to s3://gelos-fm/
│   ├── calculate_statistics.py   <- Compute dataset band statistics (hard-coded in the dataset)
│   ├── compare_model_results.py  <- Per-class random-forest accuracy comparison charts
│   ├── gelosdataset_lc.py        <- GELOSLCDataSet and its timestamp / time+location variants
│   └── plot_embeddings.py        <- Embedding plotting helper
│
└── tests
    ├── test_data.py
    ├── test_prithvi_tl.py
    └── utils.py
```

Also present locally but not tracked: `generation_logs/` (pipeline step logs),
`lightning_logs/`, `compose.override.yml`, and `gelos/` (mount point for the local gelos
checkout in the dev container).
