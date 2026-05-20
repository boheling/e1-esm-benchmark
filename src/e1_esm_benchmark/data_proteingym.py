"""ProteinGym DMS substitutions loader.

ProteinGym is a *suite* of 217 deep mutational scanning assays, each with its
own protein and its own fitness landscape. The published evaluation aggregates
a per-assay metric (Spearman) across assays. That doesn't fit the single
``(train, val, test)`` contract of the FLIP-style tasks, so it lives here.

This module returns a list of :class:`AssaySplits`, one per assay, each with
a random 80/10/10 split over that assay's mutants. Downstream code in
:mod:`benchmark_proteingym` loops over assays, embeds each assay's sequences
once per encoder (cached), fits a head, and computes per-assay Spearman.

To keep first-time runs tractable, the default loads only the first
``n_assays`` (sorted by ``DMS_id``). Override with the ``assays=`` list to
target specific ones, or set ``n_assays=None`` for the full sweep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROTEINGYM_HF_ID = "OATML-Markslab/ProteinGym_v1"
PROTEINGYM_CONFIG = "DMS_substitutions"


@dataclass
class AssaySplits:
    assay_id: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        return (f"{self.assay_id}: train={len(self.train)} val={len(self.val)} "
                f"test={len(self.test)}")


def _random_split(df: pd.DataFrame, seed: int = 0,
                  fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    tr = df.iloc[idx[:n_train]].reset_index(drop=True)
    va = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    te = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    return tr, va, te


def load_proteingym_assays(
    assays: list[str] | None = None,
    n_assays: int | None = 5,
    data_dir: str | Path = "data/proteingym",
    seed: int = 0,
    drop_ambiguous: bool = True,
) -> list[AssaySplits]:
    """Return a list of :class:`AssaySplits`, one per selected DMS assay.

    Parameters
    ----------
    assays : explicit list of ``DMS_id`` values. Wins over ``n_assays``.
    n_assays : if ``assays`` is None, take the first ``n_assays`` distinct
        DMS_ids (sorted) from the dataset. ``None`` means *all 217*.
    """
    from datasets import load_dataset

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s [%s] (this caches to %s on first run)",
                PROTEINGYM_HF_ID, PROTEINGYM_CONFIG, data_dir)
    ds = load_dataset(PROTEINGYM_HF_ID, PROTEINGYM_CONFIG,
                      cache_dir=str(data_dir), split="train")

    df = ds.to_pandas()
    expected = {"mutated_sequence", "DMS_score", "DMS_id"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ProteinGym dataset missing columns: {missing}. "
                         f"Got {df.columns.tolist()}")

    df = df.rename(columns={"mutated_sequence": "sequence", "DMS_score": "target"})
    df = df[["DMS_id", "sequence", "target"]].dropna()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["target"] = df["target"].astype(float)
    if drop_ambiguous:
        mask = ~df["sequence"].str.contains("[BJOUZ]", regex=True, na=False)
        df = df[mask]

    all_ids = sorted(df["DMS_id"].unique().tolist())
    logger.info("ProteinGym DMS_substitutions: %d assays, %d rows total",
                len(all_ids), len(df))

    if assays is not None:
        selected = [a for a in assays if a in set(all_ids)]
        unknown = set(assays) - set(all_ids)
        if unknown:
            logger.warning("Skipping %d unknown DMS_ids: %s",
                           len(unknown), sorted(unknown)[:5])
    elif n_assays is None:
        selected = all_ids
    else:
        selected = all_ids[:n_assays]

    logger.info("Loading %d assay(s): %s%s",
                len(selected), selected[:3],
                "..." if len(selected) > 3 else "")

    out: list[AssaySplits] = []
    for aid in selected:
        sub = df[df["DMS_id"] == aid][["sequence", "target"]].reset_index(drop=True)
        if len(sub) < 20:
            logger.warning("Skipping assay %s — only %d rows", aid, len(sub))
            continue
        tr, va, te = _random_split(sub, seed=seed)
        out.append(AssaySplits(assay_id=aid, train=tr, val=va, test=te))
        logger.info(out[-1].summary())
    return out
