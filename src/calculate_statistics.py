"""Calculate per-band mean and standard deviation statistics for the GELOSLCDataSet."""

import json
from pathlib import Path

from gelos.gelosdatamodule import GELOSDataModule
from loguru import logger
import torch
from tqdm import tqdm
import typer

from src.gelosdataset_lc import GELOSLCDataSet

app = typer.Typer()


def make_dataloader(
    data_root: Path,
    batch_size: int = 8,
    num_workers: int = 16,
    db_scale_bands: dict | None = None,
):
    no_norm_means = {
        sensor: {band: 0 for band in bands}
        for sensor, bands in GELOSLCDataSet.all_band_names.items()
    }
    no_norm_stds = {
        sensor: {band: 1 for band in bands}
        for sensor, bands in GELOSLCDataSet.all_band_names.items()
    }

    # Pass db_scale_bands through only when requested, so older gelos versions
    # without the parameter keep working.
    extra_kwargs = {"db_scale_bands": db_scale_bands} if db_scale_bands else {}
    datamodule = GELOSDataModule(
        data_root=data_root,
        dataset_class=GELOSLCDataSet,
        batch_size=batch_size,
        num_workers=num_workers,
        means=no_norm_means,
        stds=no_norm_stds,
        **extra_kwargs,
    )
    datamodule.setup("predict")
    loader = datamodule.predict_dataloader()
    dataset = loader.dataset
    return loader, dataset


def compute_statistics(loader, dataset, device=None):
    """Compute per-band means and standard deviations across the full dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modalities = list(dataset.bands.keys())

    # Initialize accumulators
    sums = {}
    sum_squares = {}
    pixel_counts = {}
    first_batch = True

    for batch in tqdm(loader, total=len(loader), desc="Computing statistics"):
        image_dict = batch["image"]

        for modality, tensor in image_dict.items():
            tensor = tensor.to(device)

            if first_batch:
                num_channels = tensor.shape[1]
                sums[modality] = torch.zeros(num_channels, device=device)
                sum_squares[modality] = torch.zeros(num_channels, device=device)
                _, t, _, h, w = tensor.shape
                pixel_counts[modality] = len(dataset) * t * h * w

            sums[modality] += torch.sum(tensor, dim=(0, 2, 3, 4))
            sum_squares[modality] += torch.sum(tensor.pow(2), dim=(0, 2, 3, 4))

        if first_batch:
            first_batch = False

    # Compute means and stds
    means = {m: sums[m] / pixel_counts[m] for m in modalities}
    stds = {m: torch.sqrt(sum_squares[m] / pixel_counts[m] - means[m].pow(2)) for m in modalities}

    # Format results using band names from the dataset class
    bands_per_modality = dataset.all_band_names
    formatted_means = {}
    formatted_stds = {}

    for modality in modalities:
        mean_values = means[modality].cpu().tolist()
        std_values = stds[modality].cpu().tolist()
        band_names = bands_per_modality[modality]

        formatted_means[modality] = {band: mean for band, mean in zip(band_names, mean_values)}
        formatted_stds[modality] = {band: std for band, std in zip(band_names, std_values)}

    return {"MEANS": formatted_means, "STDS": formatted_stds}


@app.command()
def main(
    data_version: str = typer.Argument(..., help="Data version subdirectory (e.g. v0.50.1)."),
    data_dir: Path = typer.Option(
        Path("/app/data/raw"), "--data-dir", "-d", help="Directory for raw data."
    ),
    output_dir: Path = typer.Option(
        Path("/app/data/interim"),
        "--output-dir",
        "-o",
        help="Root directory for processed output.",
    ),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size for the dataloader."),
    num_workers: int = typer.Option(
        16, "--num-workers", "-n", help="Number of dataloader workers."
    ),
    db_scale_bands: str = typer.Option(
        None,
        "--db-scale-bands",
        help="JSON dict of bands to convert to dB before computing statistics, "
        'e.g. \'{"S1RTC": ["VV", "VH"]}\'.',
    ),
):
    """Compute per-band means and standard deviations across the full GELOSLCDataSet."""
    data_root = data_dir / data_version
    output_path = output_dir / data_version / "statistics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data root: {data_root}")
    logger.info(f"Output path: {output_path}")

    db_scale_bands_dict = json.loads(db_scale_bands) if db_scale_bands else None
    loader, dataset = make_dataloader(data_root, batch_size, num_workers, db_scale_bands_dict)
    logger.info(f"Dataset modalities: {list(dataset.bands.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    results = compute_statistics(loader, dataset, device)

    logger.info("MEANS = " + json.dumps(results["MEANS"], indent=4))
    logger.info("STDS = " + json.dumps(results["STDS"], indent=4))

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved statistics to {output_path}")


if __name__ == "__main__":
    app()
