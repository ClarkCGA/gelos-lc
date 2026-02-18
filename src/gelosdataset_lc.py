from pathlib import Path
from typing import Any, List

import albumentations as A
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import torch

from gelos.gelosdataset import GELOSDataSet


def scale(array: np.array):
    """Scales a numpy array to 0-1 according to maximum value."""
    if array.max() > 1.0:
        array_scaled = array / 4000
    else:
        array_scaled = array * 5

    array_norm = np.clip(array_scaled, 0, 1)
    return array_norm


class GELOSLCDataSet(GELOSDataSet):
    """
    Land-cover dataset for embedding extraction and exploration.
    Contains Sentinel 1 and 2 data, DEM, and Landsat 8 and 9 data.

    Dataset Format:

    .tif files for Sentinel 1, Sentinel 2, DEM, and Landsat 8 and 9 data
    .geojson chip tracker with chip-level land cover classification

    Dataset Features:
    TBD Dataset Size
    4 time steps for each land cover chip
    """

    S2RTC_BAND_NAMES = [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
    ]
    S1RTC_BAND_NAMES = [
        "VV",
        "VH",
        # TODO 2025-10-17 GELOS v0.40 does not differentiate ASC and DSC S1 passes
        # "ASC_VV",
        # "ASC_VH",
        # "DSC_VV",
        # "DSC_VH",
        # "VV_VH",
    ]
    LANDSAT_BAND_NAMES = [
        "coastal",  # Coastal/Aerosol (Band 1)
        "blue",  # Blue (Band 2)
        "green",  # Green (Band 3)
        "red",  # Red (Band 4)
        "nir08",  # Near Infrared (NIR, Band 5)
        "swir16",  # Shortwave Infrared 1 (SWIR1, Band 6)
        "swir22",  # Shortwave Infrared 2 (SWIR2, Band 7)
    ]
    DEM_BAND_NAMES = ["DEM"]
    all_band_names = {
        "S1RTC": S1RTC_BAND_NAMES,
        "S2L2A": S2RTC_BAND_NAMES,
        "landsat": LANDSAT_BAND_NAMES,
        "DEM": DEM_BAND_NAMES,
    }

    rgb_bands = {
        "S1RTC": [],
        "S2L2A": ["RED", "GREEN", "BLUE"],
        "landsat": ["red", "green", "blue"],
        "DEM": [],
    }

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        data_root: str | Path,
        bands: dict[str, List[str]] = BAND_SETS["all"],
        means: dict[str, dict[str, float]] | None = None,
        stds: dict[str, dict[str, float]] | None = None,
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, List[str]] | None = None,
        perturb_alpha: float = 1,
    ) -> None:
        """
        Initializes an instance of GELOSLCDataSet.

        Args:
        data_root (str | Path): root directory where the dataset can be found
        means (dict[str, dict[str, float]]): Dataset means by sensor and band for scaling perturbations
        stds (dict[str, dict[str, float]]): Dataset standard deviations by sensor and band for scaling perturbations
        bands: (Dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
        transform (A.compose, optional): transform to apply. Defaults to ToTensorV2.
        concat_bands (bool, optional): concatenate all modalities into the channel dimension
        repeat_bands (dict[str, int], optional): repeat bands when loading from disc, intended to repeat single time step modalities e.g. DEM
        perturb_bands (dict[str, List[str]], optional): perturb bands with additive gaussian noise. Dictionary defining modalities and bands for perturbation.
        perturb_alpha (float, optional): relative weight given to source data vs perturbation noise. 0 signifies all noise, 1 signifies equal weights
        """
        
        super().__init__(
            bands=bands,
            all_band_names=self.all_band_names,
            means=means,
            stds=stds,
            transform=transform,
            concat_bands=concat_bands,
            repeat_bands=repeat_bands,
            perturb_bands=perturb_bands,
            perturb_alpha=perturb_alpha,
            )

        self.data_root = Path(data_root)
        self.gdf = gpd.read_file(self.data_root / "gelos_chip_tracker.geojson")
        self.zfill_length = int(self.gdf["id"].astype(str).str.len().max())
 
    def __len__(self) -> int:
        return len(self.gdf)

    def _get_file_paths(self, index: int, sensor: str) -> list[Path]:
        sample_row = self.gdf.iloc[index]
        return [
            self.data_root / filepath
            for filepath in sample_row[f"{sensor.lower()}_paths"].split(",")
        ]

    def _load_file(self, path: Path, band_indices: list[int]) -> np.ndarray:
        data = rxr.open_rasterio(path, masked=True).to_numpy()
        return data[band_indices, :, :].transpose(1, 2, 0)  # [H, W, C]

    def _get_sample_id(self, index: int) -> tuple[str, Any]:
        sample_row = self.gdf.iloc[index]
        filename = str(sample_row["id"]).zfill(self.zfill_length)
        return filename, sample_row["id"]

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        vis_bands: dict[str, dict[str, int]] = rgb_bands,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            vis_bands: bands from sensors to visualize in composites
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        if isinstance(sample["image"], dict):
            nrows = len(self.bands.keys())
        else:
            nrows = 1
        ncols = 4

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 5, nrows * 5),
            squeeze=False,
            constrained_layout=True,
        )

        if not isinstance(sample["image"], dict):
            sens = list(self.bands.keys())[0]
            sample["image"] = {sens: sample["image"]}

        for row, sens in enumerate(self.bands.keys()):
            band_indices = [self.bands[sens].index(band) for band in vis_bands[sens]]
            img = sample["image"][sens].numpy()
            img = scale(img)
            c, t, h, w = img.shape
            for col, t in enumerate(range(t)):
                img_t = img[band_indices, t, :, :].transpose(1, 2, 0)
                axs[row, col].imshow(img_t)
                axs[row, col].axis("off")

        if show_titles:
            axs[row, 0].set_title(sens)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
