"""CPU-only tests for the Prithvi EO V2 TL (time + location) experiments (issue #50).

No weight downloads: the model-dependent tests build a *random-init tiny* TL
encoder exactly as gelos' own ``tests/test_prithvi_tl_backbone.py`` does
(``prithvi_eo_v2_tiny_tl`` with ``pretrained=False, embed_dim=32, depth=2,
num_heads=2``). Its 16-px patches give 36 tokens per timestep on a 96x96 chip,
the same token layout as exp001 (``prithvi_eo_v2_300``), so the slice indices
15/36 and 73-77 used by exp038 are exercised end to end.
"""

from pathlib import Path

import pytest
import torch
import yaml
from test_data import dummy_gelos_data  # noqa: F401 — pytest fixture

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Prithvi's six HLS bands, gelos-lc names (as in exp001/exp004/exp038/exp039).
PRITHVI_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
TINY_KW = dict(pretrained=False, bands=PRITHVI_BANDS, embed_dim=32, depth=2, num_heads=2)
EMBED = 32
DEPTH = 2
N_TIMESTEPS = 4
TOKENS_PER_STEP = 36  # 96x96 chip, 16x16 patches (exp001 layout)
TL_COORDS_NAMES = ["prithvi_eo_v2_300_tl_coords", "prithvi_eo_v2_600_tl_coords"]


def _tiny(**overrides):
    from gelos.backbones.prithvi_tl_backbone import PrithviTLBackbone

    kw = {**TINY_KW, **overrides}
    torch.manual_seed(0)
    return PrithviTLBackbone("prithvi_eo_v2_tiny_tl", **kw).eval()


def _clear(model) -> None:
    model.clear_batch_timestamps()
    model.clear_batch_location()


def _time_location_batch(data_root) -> dict:
    """A real GELOSLCTimeLocationDataSet batch with exp001's transform list."""
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    from gelos.gelosdatamodule import GELOSDataModule
    from terratorch.datasets.transforms import (
        FlattenTemporalIntoChannels,
        UnflattenTemporalFromChannels,
    )

    from src.gelosdataset_lc import GELOSLCTimeLocationDataSet

    datamodule = GELOSDataModule(
        data_root=Path(data_root),
        dataset_class=GELOSLCTimeLocationDataSet,
        batch_size=1,
        num_workers=0,
        bands={"S2L2A": PRITHVI_BANDS},
        transform=A.Compose(
            [
                FlattenTemporalIntoChannels(),
                ToTensorV2(),
                UnflattenTemporalFromChannels(n_timesteps=N_TIMESTEPS),
            ]
        ),
    )
    datamodule.setup("predict")
    return next(iter(datamodule.predict_dataloader()))


# ---------------------------------------------------------------------------
# Pin guard + normalization: the `_tl_coords` names must resolve (gelos >= v0.8.0).
# ---------------------------------------------------------------------------


def test_prithvi_tl_coords_backbones_registered():
    """Pin guard: fails on gelos v0.7.0, which has no Prithvi TL wrapper."""
    import gelos.generation  # noqa: F401 — registers the `_tl_coords` factories
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

    for name in TL_COORDS_NAMES:
        assert name in TERRATORCH_BACKBONE_REGISTRY, f"{name} not registered"


@pytest.mark.parametrize("name", TL_COORDS_NAMES)
def test_normalization_resolves_for_tl_coords_names(name):
    """The `_coords` names keep the prithvi_eo_v2 prefix -> Prithvi means/stds."""
    from gelos.normalization import resolve_model_normalization

    bands = {"S2L2A": PRITHVI_BANDS}
    resolved = resolve_model_normalization(name, bands)
    plain = resolve_model_normalization(name.replace("_tl_coords", ""), bands)

    assert resolved is not None
    assert "means" in resolved and "stds" in resolved
    assert resolved["means"]["S2L2A"].keys() == set(PRITHVI_BANDS)
    assert resolved.get("set_nodata") == 0
    assert resolved == plain


# ---------------------------------------------------------------------------
# End to end on CPU: dataset batch -> wrapper (5D path, exp001 token layout).
# ---------------------------------------------------------------------------


def test_time_location_batch_drives_wrapper_on_5d_path(dummy_gelos_data):  # noqa: F811
    batch = _time_location_batch(dummy_gelos_data)
    x = batch["image"]
    assert x.shape == (1, len(PRITHVI_BANDS), N_TIMESTEPS, 96, 96)
    assert batch["timestamps"].shape == (1, N_TIMESTEPS, 3)
    assert batch["location"].shape == (1, 2)

    model = _tiny(num_frames=N_TIMESTEPS)
    assert model.num_frames == N_TIMESTEPS
    model.set_batch_timestamps(batch["timestamps"])
    model.set_batch_location(batch["location"])
    try:
        out = model(x)
    finally:
        _clear(model)

    assert isinstance(out, list)
    assert len(out) == DEPTH
    # One CLS + 4 x 36 patch tokens: the layout exp001/exp038's slice indices
    # (15/36 for the centre patch, 73-77 for the Apr-Jun block) rely on.
    assert out[-1].shape == (1, 1 + N_TIMESTEPS * TOKENS_PER_STEP, EMBED)
    # Coords actually reached the encoder (the stock silent-no-op path did not run).
    assert not torch.allclose(out[-1], model.encoder(x)[-1])


def test_task_dispatch_reaches_wrapper_without_temporal_wrapper(dummy_gelos_data, monkeypatch):  # noqa: F811
    """LenientEmbeddingGenerationTask.predict_step threads both keys to a bare
    wrapper (no TemporalWrapper, as in exp038/exp039) and clears the stashes."""
    from gelos.generation import LenientEmbeddingGenerationTask

    batch = _time_location_batch(dummy_gelos_data)
    model = _tiny(num_frames=N_TIMESTEPS)

    task = LenientEmbeddingGenerationTask.__new__(LenientEmbeddingGenerationTask)
    # The bare task skipped Module.__init__, so assigning a Module attribute via
    # the normal setattr path raises; store it directly on the instance dict.
    object.__setattr__(task, "model", model)

    def fake_super_predict_step(self, batch):
        return self.model(batch["image"])

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    result = task.predict_step(dict(batch))
    assert result[-1].shape == (1, 1 + N_TIMESTEPS * TOKENS_PER_STEP, EMBED)
    assert model._batch_timestamps is None
    assert model._batch_location is None

    no_location = {k: v for k, v in batch.items() if k != "location"}
    with pytest.raises(ValueError, match="location"):
        task.predict_step(no_location)
    assert model._batch_timestamps is None
    assert model._batch_location is None


def test_num_frames_4_equals_num_frames_1_without_coords():
    """Adding model_args.num_frames: 4 (required by the wrapper's 5D check)
    changes nothing about the encoding itself: Prithvi's positional embedding
    is a fixed sincos grid recomputed for the actual number of input frames."""
    enc_1 = _tiny(num_frames=1).encoder
    enc_4 = _tiny(num_frames=N_TIMESTEPS).encoder
    assert enc_1.num_frames == 1 and enc_4.num_frames == N_TIMESTEPS

    torch.manual_seed(1)
    x = torch.randn(1, len(PRITHVI_BANDS), N_TIMESTEPS, 32, 32)
    with torch.no_grad():
        out_1 = enc_1(x)[-1]
        out_4 = enc_4(x)[-1]
    assert out_1.shape == (1, 1 + N_TIMESTEPS * 4, EMBED)  # 32x32 -> 4 patches/step
    torch.testing.assert_close(out_1, out_4)


# ---------------------------------------------------------------------------
# Config sanity: exp038/exp039 are exp001/exp004 plus exactly the TL keys.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "baseline, tl, model_name, title",
    [
        ("exp001_prithvi300", "exp038_prithvi300_tl", "prithvi_eo_v2_300_tl_coords", "Prithvi EO V2 300M TL"),
        ("exp004_prithvi600", "exp039_prithvi600_tl", "prithvi_eo_v2_600_tl_coords", "Prithvi EO V2 600M TL"),
    ],
)
def test_tl_configs_differ_from_baselines_only_in_intended_keys(baseline, tl, model_name, title):
    base_cfg = yaml.safe_load((CONFIGS / f"{baseline}.yaml").read_text())
    tl_cfg = yaml.safe_load((CONFIGS / f"{tl}.yaml").read_text())

    assert tl_cfg["experiment_name"] == title
    assert tl_cfg["model"]["title"] == title
    assert tl_cfg["model"]["init_args"]["model"] == model_name
    assert tl_cfg["model"]["init_args"]["model_args"]["num_frames"] == N_TIMESTEPS
    assert (
        tl_cfg["data"]["init_args"]["dataset_class"]
        == "src.gelosdataset_lc.GELOSLCTimeLocationDataSet"
    )
    # 5D joint path on purpose (keeps the baseline token layout).
    assert "temporal_cfg" not in tl_cfg["model"]["init_args"]

    tl_cfg["experiment_name"] = base_cfg["experiment_name"]
    tl_cfg["model"]["title"] = base_cfg["model"]["title"]
    tl_cfg["model"]["init_args"]["model"] = base_cfg["model"]["init_args"]["model"]
    del tl_cfg["model"]["init_args"]["model_args"]["num_frames"]
    tl_cfg["data"]["init_args"]["dataset_class"] = base_cfg["data"]["init_args"]["dataset_class"]
    assert tl_cfg == base_cfg
