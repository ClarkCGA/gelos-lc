# GELOS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for Geospatial Exploration of Latent Observation Space (GELOS)

## Sample Commands

Running generation for all configs in a directory in parallel, using 8 GPUs:
```
docker compose run --rm prod scripts/run_embeddings.sh /app/configs/*
```

Running analysis for all configs in the default configs/ directory:
```
docker compose run --rm prod make analysis
```


## Timestamp experiments (OlmoEarth v1.2, issue #47)

Experiments exp024–exp031 never override the gelos dataset metadata hooks, so the
OlmoEarth backbone feeds the constant dummy date `[15, 0, 2020]` (all four
timesteps = January 2020) to every chip. `src.gelosdataset_lc.GELOSLCMetadataDataSet`
(gelos >= v0.6.0, hooks from gelos#79; the project pins gelos v0.7.0) returns each chip's real Sentinel-2
acquisition dates from the chip tracker's `s2l2a_dates` column; configs opt in with
`data.init_args.dataset_class: src.gelosdataset_lc.GELOSLCMetadataDataSet`.

| Config | Baseline | Difference |
|---|---|---|
| `exp036_olmoearth_v1_2_base_s2_timestamps` | `exp030` (V1.2 Base S2) | real S2 dates |
| `exp037_olmoearth_v1_2_base_s2_s1_timestamps` | `exp031` (V1.2 Base S2+S1) | real S2 dates |

Comparisons: `configs/comparisons/16_olmoearth_timestamps_knn.yaml` (kNN purity),
`17a_*`/`17b_*` (per-chip cosine similarity to the generic-date control) and
`make compare-model-results` (downstream random-forest accuracy per class,
`src/compare_model_results.py`); figures under `reports/figures/comparisons/`.

Caveats for anyone reading this for issue #23 ("give every model every modality it
was trained to accept"):

- OlmoEarth uses only the **month** of each timestamp (olmoearth_pretrain
  `nn/flexi_vit.py`, `months = timestamps[:, :, 1]`); day and year are ignored, so
  these runs measure real-month vs all-January seasonality encoding only.
- **Location is not threaded.** gelos exposes a `_get_location` hook, but no gelos
  backbone consumes it (OlmoEarth's encoder never reads `latlon`), so it was left out
  rather than adding inert plumbing.

Findings (v0.50.1, center patch, all 4 time steps, layer_0):

- **Embedding space moves a lot.** Per-chip cosine similarity to the generic-date
  control averages 0.79–0.85 per class for S2 (`17a`) and 0.83–0.89 for S2+S1
  (`17b`), KS p = 0 for every class; the mean-embedding cosine similarity between
  control and real-date runs is 0.86 (S2). The month embedding is clearly reaching
  the encoder (there is no "falling back to the constant date" warning in the
  generation logs).
- **Downstream separability barely changes.** Overall kNN purity is within 0.1 pp at
  every k (e.g. k=10: 0.9742 vs 0.9734 for S2, 0.9730 vs 0.9737 for S2+S1) and
  overall random-forest accuracy is 0.9745 vs 0.9746 (S2) and 0.9734 vs 0.9745
  (S2+S1). Per class the largest shifts are Built Area (−1.4 to −1.6 pp) and Bare
  Ground (+0.2 / +1.1 pp); everything else is within ±0.3 pp.

Takeaway: for chip-level land-cover class separability, correct seasonality
encoding is not a meaningful lever for OlmoEarth v1.2 on this dataset, even though
it changes the embeddings themselves substantially.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         gelos and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes this directory Python module
    │
    ├── app_files_generation.py          <- Generate all Gelos App files (json, pmtiles, config.js)
    │
    ├── app_files_upload.py              <- Upload all Gelos App files to s3://gelos-fm/
``` │ 
    └── gelosdataset_lc.py                  <- GELOS LC Dataset module
    --------

