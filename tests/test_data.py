from shapely import Polygon
import pdb
import os
import gc
import pandas as pd
import geopandas as gpd
import pytest
from torch.utils.data import DataLoader
from utils import create_dummy_image
from pathlib import Path
import torch
@pytest.fixture
def dummy_gelos_data(tmp_path) -> str:
    base_dir = tmp_path / "gelos"
    base_dir.mkdir()
    metadata_filename = "gelos_chip_tracker.geojson"
    metadata_path = base_dir / metadata_filename
    
    # Create a GeoDataFrame that matches the GeoJSON structure
    data = {
        "id": [0],
        "s2l2a_dates": ["20230218,20230419,20230713,20231230"],
        "s1rtc_dates": ["20230218,20230419,20230712,20231227"],
        "lc2l2_dates": ["20230217,20230524,20230921,20231218"],
        "s2l2a_paths": ["s2l2a_000000_20230218.tif,s2l2a_000000_20230419.tif,s2l2a_000000_20230713.tif,s2l2a_000000_20231230.tif"],
        "s1rtc_paths": ["s1rtc_000000_20230218.tif,s1rtc_000000_20230419.tif,s1rtc_000000_20230712.tif,s1rtc_000000_20231227.tif"],
        "lc2l2_paths": ["lc2l2_000000_20230217.tif,lc2l2_000000_20230524.tif,lc2l2_000000_20230921.tif,lc2l2_000000_20231218.tif"],
        "dem_paths": ["dem_000000.tif"],
        "lulc": [2],
        # chip centre in decimal degrees, as in the real tracker (used by
        # GELOSLCTimeLocationDataSet / Prithvi TL)
        "lat": [4.2856],
        "lon": [21.8256],
    }
    for s2l2a_dates, id in zip(data['s2l2a_dates'], data['id']):
        for date in s2l2a_dates.split(','):
            create_dummy_image(base_dir / f"s2l2a_{id:06}_{date}.tif", (96, 96, 13), range(255))
    for lc2l2_dates, id in zip(data['lc2l2_dates'], data['id']):
        for date in lc2l2_dates.split(','):
            create_dummy_image(base_dir / f"lc2l2_{id:06}_{date}.tif", (96, 96, 7), range(255))
    for s1rtc_dates, id in zip(data['s1rtc_dates'], data['id']):
        for date in s1rtc_dates.split(','):
            create_dummy_image(base_dir / f"s1rtc_{id:06}_{date}.tif", (96, 96, 7), range(255))
    for id in data['id']:
        create_dummy_image(base_dir / f"dem_{id:06}.tif", (96, 96), range(255))

    # Create a dummy polygon geometry
    polygon = Polygon([
        (21.8299, 4.2812), (21.8299, 4.2899), 
        (21.8212, 4.2899), (21.8212, 4.2812), 
        (21.8299, 4.2812)
    ])
    
    gdf = gpd.GeoDataFrame(data, geometry=[polygon], crs="EPSG:4326")
    gdf.to_file(metadata_path, driver='GeoJSON')
    
    return str(base_dir)

def test_gelos_datamodule(dummy_gelos_data):
    from gelos.gelosdatamodule import GELOSDataModule
    from src.gelosdataset_lc import GELOSLCDataSet
    dummy_gelos_data = Path(dummy_gelos_data)
    batch_size = 1
    num_workers = 0
    # all bands
    datamodule = GELOSDataModule(
        data_root=dummy_gelos_data,
        dataset_class=GELOSLCDataSet,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule.setup("predict")
    predict_loader: DataLoader = datamodule.predict_dataloader()
    batch = next(iter(predict_loader))
    assert "S1RTC" in batch['image'], "Key S1 not found on predict_dataloader"
    assert "S2L2A" in batch['image'], "Key S2 not found on predict_dataloader"
    
    gc.collect()

def test_output_contract(dummy_gelos_data):
    """Batch output must contain image, filename, and file_id keys."""
    from gelos.gelosdatamodule import GELOSDataModule
    from src.gelosdataset_lc import GELOSLCDataSet

    dummy_gelos_data = Path(dummy_gelos_data)
    datamodule = GELOSDataModule(
        data_root=dummy_gelos_data,
        dataset_class=GELOSLCDataSet,
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))

    assert "image" in batch
    assert "filename" in batch
    assert "file_id" in batch
    # image should be a dict of tensors for multi-sensor
    assert isinstance(batch["image"], dict)
    for sensor_tensor in batch["image"].values():
        assert isinstance(sensor_tensor, torch.Tensor)

    gc.collect()


def test_single_sensor(dummy_gelos_data):
    """Single-sensor config should produce a plain Tensor for image."""
    from gelos.gelosdatamodule import GELOSDataModule
    from src.gelosdataset_lc import GELOSLCDataSet

    dummy_gelos_data = Path(dummy_gelos_data)
    datamodule = GELOSDataModule(
        data_root=dummy_gelos_data,
        dataset_class=GELOSLCDataSet,
        batch_size=1,
        num_workers=0,
        bands={"S2L2A": GELOSLCDataSet.S2RTC_BAND_NAMES},
    )
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))

    assert isinstance(batch["image"], torch.Tensor)

    gc.collect()


def test_clip_range_bands_forwarded_to_gelos(dummy_gelos_data):
    """GELOSLCDataSet forwards clip_range_bands to GELOSDataSet (gelos >= v0.7.0).

    The DINOv3 configs (exp034/exp035) rely on this: listed bands are clipped
    to [min, max] at load time, unlisted bands are left untouched. Dummy chips
    hold values 0..254, so clipping RED at 100 is observable.
    """
    from src.gelosdataset_lc import GELOSLCDataSet

    bands = {"S2L2A": ["RED", "GREEN", "BLUE"]}
    dataset = GELOSLCDataSet(
        data_root=dummy_gelos_data,
        bands=bands,
        transform=None,
        clip_range_bands={"S2L2A": {"RED": [0.0, 100.0]}},
    )
    assert dataset.clip_range_bands == {"S2L2A": {"RED": [0.0, 100.0]}}

    image = dataset[0]["image"]  # (C, T, H, W) for a single sensor
    assert image[0].max() <= 100.0, "RED should be clipped to the configured max"
    assert image[1].max() > 100.0, "GREEN must not be clipped"
    assert image[2].max() > 100.0, "BLUE must not be clipped"

    gc.collect()


def test_lowercase_stat_aliases():
    """GELOSDataModule resolves stats via lowercase class attributes; the
    aliases must exist and be identical to the primary uppercase dicts."""
    from src.gelosdataset_lc import GELOSLCDataSet

    assert GELOSLCDataSet.means is GELOSLCDataSet.MEANS
    assert GELOSLCDataSet.stds is GELOSLCDataSet.STDS


def test_datamodule_resolves_dataset_stats(dummy_gelos_data):
    """Instantiating the datamodule with GELOSLCDataSet must pick up the
    dataset-class statistics instead of the 0/1 identity defaults."""
    from gelos.gelosdatamodule import GELOSDataModule
    from src.gelosdataset_lc import GELOSLCDataSet

    datamodule = GELOSDataModule(
        data_root=Path(dummy_gelos_data),
        dataset_class=GELOSLCDataSet,
        batch_size=1,
        num_workers=0,
    )

    for modality, band_means in datamodule.means.items():
        assert any(
            float(mean) != 0.0 for mean in band_means
        ), f"{modality} means are all identity defaults — stats were not resolved"
    for modality, band_stds in datamodule.stds.items():
        assert any(
            float(std) != 1.0 for std in band_stds
        ), f"{modality} stds are all identity defaults — stats were not resolved"

    gc.collect()


# Expected [year, month, day] rows for the fixture's s2l2a_dates string.
EXPECTED_S2_TIMESTAMPS = [[2023, 2, 18], [2023, 4, 19], [2023, 7, 13], [2023, 12, 30]]


def test_parse_tracker_dates():
    import numpy as np
    from src.gelosdataset_lc import parse_tracker_dates

    dates = parse_tracker_dates("20230218,20230419,20230713,20231230")
    assert dates.shape == (4, 3)
    assert dates.dtype == np.int64
    assert dates.tolist() == EXPECTED_S2_TIMESTAMPS

    with pytest.raises(ValueError):
        parse_tracker_dates("20230218,2023-04-19")
    with pytest.raises(ValueError):
        parse_tracker_dates("2023021,20230419")


def _metadata_batch(data_root, dataset_class):
    from gelos.gelosdatamodule import GELOSDataModule
    from src.gelosdataset_lc import GELOSLCDataSet

    datamodule = GELOSDataModule(
        data_root=Path(data_root),
        dataset_class=dataset_class,
        batch_size=1,
        num_workers=0,
        bands={"S2L2A": GELOSLCDataSet.S2RTC_BAND_NAMES},
    )
    datamodule.setup("predict")
    return next(iter(datamodule.predict_dataloader()))


def test_metadata_dataset_batch_has_timestamps(dummy_gelos_data):
    """GELOSLCMetadataDataSet adds a (B, T, 3) long tensor of real S2 dates."""
    from src.gelosdataset_lc import GELOSLCMetadataDataSet

    batch = _metadata_batch(dummy_gelos_data, GELOSLCMetadataDataSet)

    assert "image" in batch
    assert "filename" in batch
    assert "file_id" in batch
    assert "timestamps" in batch
    assert batch["timestamps"].shape == (1, 4, 3)
    assert batch["timestamps"].dtype == torch.long
    assert batch["timestamps"][0].tolist() == EXPECTED_S2_TIMESTAMPS
    # The metadata class deliberately emits timestamps only (exp036/exp037
    # batches stay unchanged); location lives in GELOSLCTimeLocationDataSet
    # (Prithvi TL, issue #50).
    assert "location" not in batch

    gc.collect()


def test_metadata_dataset_resolves_from_string(dummy_gelos_data):
    """The YAML `dataset_class` string path resolves to the metadata dataset."""
    batch = _metadata_batch(dummy_gelos_data, "src.gelosdataset_lc.GELOSLCMetadataDataSet")

    assert "timestamps" in batch
    assert batch["timestamps"].shape == (1, 4, 3)
    assert batch["timestamps"][0].tolist() == EXPECTED_S2_TIMESTAMPS

    gc.collect()


def test_base_dataset_omits_metadata_keys(dummy_gelos_data):
    """Baseline GELOSLCDataSet behaviour is unchanged: no metadata keys."""
    from src.gelosdataset_lc import GELOSLCDataSet

    batch = _metadata_batch(dummy_gelos_data, GELOSLCDataSet)

    assert "timestamps" not in batch
    assert "location" not in batch

    gc.collect()


def test_metadata_dataset_missing_columns_raises(dummy_gelos_data):
    """A tracker without s2l2a_dates fails fast with a clear error."""
    from src.gelosdataset_lc import GELOSLCMetadataDataSet

    tracker_path = Path(dummy_gelos_data) / "gelos_chip_tracker.geojson"
    gdf = gpd.read_file(tracker_path).drop(columns=["s2l2a_dates"])
    gdf.to_file(tracker_path, driver="GeoJSON")

    with pytest.raises(ValueError, match="s2l2a_dates"):
        GELOSLCMetadataDataSet(data_root=dummy_gelos_data)


def test_time_location_dataset_batch_has_location(dummy_gelos_data):
    """GELOSLCTimeLocationDataSet adds (B, 2) float32 [lat, lon] next to timestamps."""
    from src.gelosdataset_lc import GELOSLCTimeLocationDataSet

    batch = _metadata_batch(dummy_gelos_data, GELOSLCTimeLocationDataSet)

    assert "timestamps" in batch
    assert batch["timestamps"].shape == (1, 4, 3)
    assert batch["timestamps"][0].tolist() == EXPECTED_S2_TIMESTAMPS
    assert "location" in batch
    assert batch["location"].shape == (1, 2)
    assert batch["location"].dtype == torch.float32
    assert batch["location"][0].tolist() == pytest.approx([4.2856, 21.8256])

    gc.collect()


def test_time_location_dataset_resolves_from_string(dummy_gelos_data):
    """The YAML `dataset_class` string path resolves to the time+location dataset."""
    batch = _metadata_batch(dummy_gelos_data, "src.gelosdataset_lc.GELOSLCTimeLocationDataSet")

    assert batch["timestamps"].shape == (1, 4, 3)
    assert batch["location"].shape == (1, 2)
    assert batch["location"][0].tolist() == pytest.approx([4.2856, 21.8256])

    gc.collect()


def test_time_location_dataset_missing_columns_raises(dummy_gelos_data):
    """A tracker without lat/lon fails fast with a clear error."""
    from src.gelosdataset_lc import GELOSLCTimeLocationDataSet

    tracker_path = Path(dummy_gelos_data) / "gelos_chip_tracker.geojson"
    gdf = gpd.read_file(tracker_path).drop(columns=["lat"])
    gdf.to_file(tracker_path, driver="GeoJSON")

    with pytest.raises(ValueError, match="lat"):
        GELOSLCTimeLocationDataSet(data_root=dummy_gelos_data)

