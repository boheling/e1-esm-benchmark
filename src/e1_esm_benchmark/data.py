"""Data loaders for the benchmark tasks.

All loaders return a :class:`Splits` with ``train`` / ``val`` / ``test``
DataFrames, each containing the columns ``sequence`` (str) and ``target``
(float). Downstream code in :mod:`benchmark` is task-agnostic and routes via
the :data:`TASKS` registry.

Tasks
-----
- ``meltome``      — FLIP Meltome mixed split (thermostability regression).
- ``gb1``          — FLIP GB1 sampled split (binding fitness, 4-position mut).
- ``fluorescence`` — TAPE GFP variants (proteinea/fluorescence on HF).

ProteinGym is its own thing: it's a *suite* of 217 assays with macro-Spearman
aggregation, which doesn't fit the single ``(train, val, test)`` contract.
See :mod:`data_proteingym` for its loader.
"""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Splits:
    """Train/val/test frames. Each has columns: sequence (str), target (float)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    task: str = "unknown"

    def summary(self) -> str:
        return (
            f"Splits(task={self.task}, train={len(self.train)}, "
            f"val={len(self.val)}, test={len(self.test)})"
        )


# Back-compat alias — old callers used ``MeltomeSplits``.
MeltomeSplits = Splits


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Non-canonical residues that ESM2 doesn't have dedicated tokens for. We strip
# rows containing these so both models see the same alphabet.
_AMBIGUOUS_RESIDUES = "BJOUZ"


def _download(url: str, dest: Path, desc: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached download at %s", dest)
        return
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))


def _sanitize(df: pd.DataFrame, drop_ambiguous: bool = True) -> pd.DataFrame:
    """Drop rows with missing/non-canonical sequences; uppercase + strip."""
    df = df.dropna(subset=["sequence", "target"]).copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["target"] = df["target"].astype(float)
    if drop_ambiguous:
        mask = ~df["sequence"].str.contains(f"[{_AMBIGUOUS_RESIDUES}]", regex=True, na=False)
        dropped = (~mask).sum()
        if dropped:
            logger.info("Dropping %d sequences with ambiguous residues (%s)",
                        dropped, _AMBIGUOUS_RESIDUES)
        df = df[mask]
    return df.reset_index(drop=True)


def _carve_val(train_df: pd.DataFrame, frac: float = 0.1, seed: int = 0
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation slice out of train when the upstream CSV has none."""
    val_df = train_df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    new_train = train_df.drop(val_df.index, errors="ignore").reset_index(drop=True)
    return new_train, val_df


# ---------------------------------------------------------------------------
# FLIP Meltome
# ---------------------------------------------------------------------------

FLIP_MELTOME_URL = os.environ.get(
    "FLIP_MELTOME_URL",
    "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/meltome/splits.zip",
)


def _prepare_flip_csv(url: str, data_dir: Path, expected_csv: Path, archive_name: str,
                     desc: str) -> Path:
    if expected_csv.exists():
        return expected_csv
    zip_path = data_dir / archive_name
    _download(url, zip_path, desc=desc)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    if not expected_csv.exists():
        raise FileNotFoundError(
            f"Expected {expected_csv} after extracting {zip_path}. "
            "Upstream FLIP repo may have restructured files."
        )
    return expected_csv


def load_meltome(
    data_dir: str | Path = "data/flip_meltome",
    max_length: int | None = 1022,
    drop_ambiguous: bool = True,
) -> Splits:
    """FLIP Meltome mixed split — thermostability (°C) regression."""
    data_dir = Path(data_dir)
    csv_path = _prepare_flip_csv(
        FLIP_MELTOME_URL, data_dir,
        expected_csv=data_dir / "splits" / "mixed_split.csv",
        archive_name="meltome_splits.zip",
        desc="meltome",
    )
    df = pd.read_csv(csv_path)
    expected = {"sequence", "target", "set", "validation"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Meltome CSV missing columns: {missing}. Got {df.columns.tolist()}")

    df = _sanitize(df, drop_ambiguous=drop_ambiguous)
    val_mask = df["validation"].astype(str).str.lower().isin({"true", "1", "1.0"})
    train_df = df[(df["set"] == "train") & ~val_mask][["sequence", "target"]].reset_index(drop=True)
    val_df = df[(df["set"] == "train") & val_mask][["sequence", "target"]].reset_index(drop=True)
    test_df = df[df["set"] == "test"][["sequence", "target"]].reset_index(drop=True)

    if len(val_df) == 0 and len(train_df) > 0:
        logger.warning("No validation rows flagged in CSV; carving 10%% of train as val.")
        train_df, val_df = _carve_val(train_df)

    if max_length is not None:
        pre = len(train_df) + len(val_df)
        train_df = train_df[train_df["sequence"].str.len() <= max_length].reset_index(drop=True)
        val_df = val_df[val_df["sequence"].str.len() <= max_length].reset_index(drop=True)
        logger.info("Length-filtered train+val at <= %d residues: %d -> %d",
                    max_length, pre, len(train_df) + len(val_df))

    splits = Splits(train=train_df, val=val_df, test=test_df, task="flip_meltome_mixed")
    logger.info(splits.summary())
    return splits


# ---------------------------------------------------------------------------
# FLIP GB1
# ---------------------------------------------------------------------------

# Five FLIP GB1 splits exist; `sampled` is the standard reported one (random
# split over the 4-position combinatorial library; ~150k variants of ~56 aa).
# Others test extrapolation regimes: low_vs_high trains on low-fitness variants
# and tests on high, etc. Override `split_name` to use those.
FLIP_GB1_URL = os.environ.get(
    "FLIP_GB1_URL",
    "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/splits.zip",
)
GB1_VALID_SPLITS = ("sampled", "one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high")


def load_gb1(
    data_dir: str | Path = "data/flip_gb1",
    split_name: str = "sampled",
    drop_ambiguous: bool = True,
) -> Splits:
    """FLIP GB1 — binding fitness regression over 4-position mutants.

    Parameters
    ----------
    split_name : one of ``sampled`` (default, random split), ``one_vs_rest``,
        ``two_vs_rest``, ``three_vs_rest``, or ``low_vs_high``. The non-sampled
        splits are extrapolation regimes — useful follow-ups, but `sampled` is
        what the FLIP paper reports head-to-head.
    """
    if split_name not in GB1_VALID_SPLITS:
        raise ValueError(f"Unknown GB1 split {split_name!r}. Choose from {GB1_VALID_SPLITS}.")
    data_dir = Path(data_dir)
    csv_path = _prepare_flip_csv(
        FLIP_GB1_URL, data_dir,
        expected_csv=data_dir / "splits" / f"{split_name}.csv",
        archive_name="gb1_splits.zip",
        desc=f"gb1/{split_name}",
    )
    df = pd.read_csv(csv_path)
    expected = {"sequence", "target", "set"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"GB1 CSV missing columns: {missing}. Got {df.columns.tolist()}")

    df = _sanitize(df, drop_ambiguous=drop_ambiguous)
    train_full = df[df["set"] == "train"][["sequence", "target"]].reset_index(drop=True)
    test_df = df[df["set"] == "test"][["sequence", "target"]].reset_index(drop=True)
    train_df, val_df = _carve_val(train_full)

    splits = Splits(train=train_df, val=val_df, test=test_df, task=f"flip_gb1_{split_name}")
    logger.info(splits.summary())
    return splits


# ---------------------------------------------------------------------------
# TAPE Fluorescence (GFP)
# ---------------------------------------------------------------------------

def load_fluorescence(
    data_dir: str | Path = "data/fluorescence",
    drop_ambiguous: bool = True,
) -> Splits:
    """TAPE Fluorescence — log-fluorescence regression over GFP variants.

    Pulled from ``proteinea/fluorescence`` on HuggingFace. ~54k sequences,
    each ~236 residues, with 0–3 point mutations from the wildtype.
    """
    from datasets import load_dataset  # local import — datasets is heavy

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("proteinea/fluorescence", cache_dir=str(data_dir))

    def _as_df(split: str) -> pd.DataFrame:
        d = ds[split].to_pandas()
        out = pd.DataFrame(
            {"sequence": d["primary"], "target": d["log_fluorescence"].astype(float)}
        )
        return _sanitize(out, drop_ambiguous=drop_ambiguous)

    splits = Splits(
        train=_as_df("train"),
        val=_as_df("validation"),
        test=_as_df("test"),
        task="tape_fluorescence",
    )
    logger.info(splits.summary())
    return splits


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: dict[str, callable] = {
    "meltome": load_meltome,
    "gb1": load_gb1,
    "fluorescence": load_fluorescence,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    splits = load_meltome()
    print(splits.summary())
    print(splits.train.head())
