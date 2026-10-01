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
(gelos >= v0.6.0, hooks from gelos#79; the project pins gelos v0.8.0, see Dockerfile) returns each chip's real Sentinel-2
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
- **Location is not threaded for OlmoEarth** (its encoder never reads `latlon`), so
  `GELOSLCMetadataDataSet` emits timestamps only rather than inert plumbing. Location
  *is* threaded for Prithvi TL — see the next section.

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

## Time + location experiments (Prithvi EO V2 TL, issue #50)

Prithvi EO V2 ships "TL" checkpoints (300M-TL, 600M-TL) whose encoder adds two
sinusoidal embeddings (each with a learned scale) to every patch token: a
**temporal** one from `[year, day-of-year]` per timestep and a **location** one from
`[lat, lon]` per chip. Baselines exp001/exp004 run the plain checkpoints, which have
neither.

- **Backbone.** gelos v0.8.0 exposes the TL encoders through a wrapper,
  `gelos/backbones/prithvi_tl_backbone.py` (gelos#78), registered as
  `prithvi_eo_v2_{300,600}_tl_coords`. Do not use terratorch's plain
  `prithvi_eo_v2_*_tl` names: they load the TL weights, but the embedding task calls
  the backbone with the image alone, so the time/location embeddings are silently
  dropped. The `_coords` wrapper takes `batch["timestamps"]` / `batch["location"]`
  from the gelos side-channel, converts the dates to `[year, doy]`, forwards both as
  `temporal_coords` / `location_coords`, and **raises** if either key is missing.
- **Dataset.** `src.gelosdataset_lc.GELOSLCTimeLocationDataSet` extends
  `GELOSLCMetadataDataSet` with `_get_location` from the chip tracker's `lat`/`lon`
  columns (chip centre, decimal degrees), so each batch carries `timestamps`
  `(B, T, 3)` (from `s2l2a_dates`) and `location` `(B, 2)`. Configs opt in with
  `data.init_args.dataset_class: src.gelosdataset_lc.GELOSLCTimeLocationDataSet`.
- **`model_args.num_frames: 4`, no `temporal_cfg`.** Like exp001/exp004 the TL configs
  feed the 5D `(B, 6, 4, H, W)` chip straight to the encoder (joint space-time
  encoding), which keeps the one-CLS + 4x36 (600M: 4x49) token layout so the slice
  indices line up with the baselines. The wrapper's 5D path requires
  `T == num_frames`, and terratorch defaults `num_frames=1`; the value does not
  change the encoding otherwise (`tests/test_prithvi_tl.py` pins this). The
  upstream-recommended `temporal_cfg.temporal_wrapper: true` would change the token
  layout and is deliberately not used.

| Config | Baseline | Difference |
|---|---|---|
| `exp038_prithvi300_tl` | `exp001` (Prithvi 300M) | 300M-TL checkpoint + real S2 dates + chip lat/lon |
| `exp039_prithvi600_tl` | `exp004` (Prithvi 600M) | 600M-TL checkpoint + real S2 dates + chip lat/lon |

Comparisons: `configs/comparisons/18_prithvi_tl_knn.yaml` (four-way kNN purity with
paired violins), `19a_prithvi300_tl_embeddings.yaml` / `19b_prithvi600_tl_embeddings.yaml`
(per-chip cosine similarity to the no-coords control) and
`make compare-model-results-prithvi-tl` (downstream random-forest accuracy per class).
gelos v0.8.0 writes figures per config stem, so they land under
`reports/figures/comparisons/{18_prithvi_tl_knn,19a_prithvi300_tl_embeddings,19b_prithvi600_tl_embeddings}/`
plus `reports/figures/comparisons/18_prithvi_tl/` for the random-forest chart, and the
experiment figures under `reports/figures/v0.50.1/exp038_prithvi300_tl/` and
`.../exp039_prithvi600_tl/`.

Caveat: the TL checkpoints are separately trained weights, so unlike 17a/17b (same
OlmoEarth weights, dates on/off) 19a/19b measure checkpoint **and** coords jointly —
the full "TL treatment", not a pure coords ablation.

Findings: _to be filled in after the exp038/exp039 runs._

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

