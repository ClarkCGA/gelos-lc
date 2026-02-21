#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path

import geopandas as gpd
import subprocess
import yaml

from gelos.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_VERSION, INTERIM_DATA_DIR


# set directory names and creat an app files directory
output_dir = PROCESSED_DATA_DIR / DATA_VERSION
interim_dir = INTERIM_DATA_DIR / DATA_VERSION
interim_dir.mkdir(exist_ok=True, parents=True)
app_files_dir = output_dir / "app_files"
app_files_dir.mkdir(exist_ok=True, parents=True)
json_dir = app_files_dir / "json"
json_dir.mkdir(exist_ok=True, parents=True)
pmtiles_dir = app_files_dir / "pmtiles"
pmtiles_dir.mkdir(exist_ok=True, parents=True)
# get a list of all embedding csv paths
embedding_csv_paths = output_dir.rglob("*tsne.csv")


# get the original chip tracker from the raw data directory
data_root = RAW_DATA_DIR / DATA_VERSION
chip_gdf = gpd.read_file(data_root / 'gelos_chip_tracker.geojson')

# get only the fields we need for pmtiles
chip_gdf = chip_gdf[[
    "category",
    "sentinel_1_dates",
    "sentinel_2_dates",
    "landsat_dates",
    "id",
    "lat",
    "lon",
    "color",
    "landsat_thumbs",
    "sentinel_1_thumbs",
    "sentinel_2_thumbs",
    "geometry",
    ]]


# for the points.json, drop geometry to make the file smaller and faster to load in the browser
points_df = chip_gdf.drop("geometry", axis=1)
points_df.to_json(json_dir / "points.json")


# copy all embedding files to json in the app files directory
for embedding_csv_path in embedding_csv_paths:
    embed_df = gpd.read_file(embedding_csv_path)
    embed_df = embed_df.rename(columns={"dim_1":"x", "dim_2":"y"})
    embed_df.to_json(json_dir / f"{embedding_csv_path.stem}.json")


# save centroids and chips to temporary files for tippecanoe to load from
chip_gdf_chip_tracker = chip_gdf[[
    "category",
    "id",
    "color",
    "geometry",
    ]]
chip_gdf_centroids = chip_gdf_chip_tracker.copy()
chip_gdf_centroids["geometry"] = chip_gdf_centroids.geometry.centroid
chip_gdf_centroids = chip_gdf_centroids.set_geometry("geometry")
chip_gdf_centroids.to_file(interim_dir / "gelos_app_centroids.geojson", driver="GeoJSON")
chip_gdf_chip_tracker.to_file(interim_dir / "gelos_app_chip_tracker.geojson", driver="GeoJSON")


# generate pmtiles using tippecanoe
cmd = f"""
tippecanoe -zg \
  -l gelos_centroids \
  -o {str(pmtiles_dir / "centroids.pmtiles")} \
  {str(interim_dir / "gelos_app_centroids.geojson")}
"""

subprocess.run(cmd, shell=True, check=True)

cmd = f"""
tippecanoe -f \
  -ps \
  --no-tiny-polygon-reduction \
  --no-tile-size-limit \
  --no-feature-limit \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  -l gelos_chips \
  -o {str(pmtiles_dir / "gelos_chip_tracker.pmtiles")} \
  {str(interim_dir / "gelos_app_chip_tracker.geojson")}
"""


subprocess.run(cmd, shell=True, check=True)


# generate models.json from experiment config YAMLs
S3_JSON_BASE_URL = "https://gelos-fm.s3.amazonaws.com/json"
configs_dir = Path(__file__).parent.parent / "configs"
models = {}

for config_path in sorted(configs_dir.glob("*.yaml")):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config_stem = config_path.stem
    experiment_name = config["experiment_name"]

    for strategy_key, strategy_config in config.get("embedding_extraction_strategies", {}).items():
        key = f"{config_stem}_{strategy_key}"
        title = f"{experiment_name}: {strategy_config['title']}"
        models[key] = {
            "path": f"{S3_JSON_BASE_URL}/{key}.json",
            "title": title,
        }

with open(json_dir / "models.json", "w") as f:
    json.dump(models, f, indent=2)

