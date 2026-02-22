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

