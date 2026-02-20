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

    MEANS = {
        "S1RTC": {
            "VV": 0.23718494176864624,
            "VH": 0.05088871344923973
        },
        "S2L2A": {
            "COASTAL_AEROSOL": 609.0277099609375,
            "BLUE": 682.1007690429688,
            "GREEN": 797.3997802734375,
            "RED": 902.5363159179688,
            "RED_EDGE_1": 1012.9698486328125,
            "RED_EDGE_2": 1170.142822265625,
            "RED_EDGE_3": 1244.34375,
            "NIR_BROAD": 1272.984619140625,
            "NIR_NARROW": 1292.264404296875,
            "WATER_VAPOR": 1296.0284423828125,
            "SWIR_1": 1293.037109375,
            "SWIR_2": 1112.1793212890625
        },
        "LC2L2": {
            "coastal": 0.04326515644788742,
            "blue": 0.052140623331069946,
            "green": 0.07505789399147034,
            "red": 0.09055962413549423,
            "nir08": 0.15706466138362885,
            "swir16": 0.15178132057189941,
            "swir22": 0.11629538238048553
        },
        "DEM": {
            "DEM": 690.3432006835938
        }
    }

    STDS = {
        "S1RTC": {
            "VV": 2.3691890239715576,
            "VH": 0.24191336333751678
        },
        "S2L2A": {
            "COASTAL_AEROSOL": 1071.2337646484375,
            "BLUE": 1176.518310546875,
            "GREEN": 1331.1412353515625,
            "RED": 1525.9931640625,
            "RED_EDGE_1": 1654.466796875,
            "RED_EDGE_2": 1814.0484619140625,
            "RED_EDGE_3": 1911.949951171875,
            "NIR_BROAD": 1953.797607421875,
            "NIR_NARROW": 1970.2528076171875,
            "WATER_VAPOR": 1976.5057373046875,
            "SWIR_1": 2027.414306640625,
            "SWIR_2": 1798.0020751953125
        },
        "LC2L2": {
            "coastal": 0.10800547897815704,
            "blue": 0.11215518414974213,
            "green": 0.1203065812587738,
            "red": 0.13768944144248962,
            "nir08": 0.16932472586631775,
            "swir16": 0.1689019352197647,
            "swir22": 0.14119330048561096
        },
        "DEM": {
            "DEM": 703.6107177734375
        }
    }

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
    ]
    LC2L2_BAND_NAMES = [
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
        "LC2L2": LC2L2_BAND_NAMES,
        "DEM": DEM_BAND_NAMES,
    }

    rgb_bands = {
        "S1RTC": [],
        "S2L2A": ["RED", "GREEN", "BLUE"],
        "LC2L2": ["red", "green", "blue"],
        "DEM": [],
    }

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        data_root: str | Path,
        bands: dict[str, List[str]] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, List[str]] | None = None,
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
        """
        
        super().__init__(
            bands=bands,
            all_band_names=self.all_band_names,
            transform=transform,
            concat_bands=concat_bands,
            repeat_bands=repeat_bands,
            perturb_bands=perturb_bands,
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
