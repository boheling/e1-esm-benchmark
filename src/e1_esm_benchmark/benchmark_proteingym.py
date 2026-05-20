"""ProteinGym benchmark orchestrator.

For each selected DMS assay:

1. Concatenate train+val+test sequences.
2. Embed (cached per-assay-per-model).
3. Slice the embedding matrix into the three splits.
4. Fit head on train (val for alpha/early stopping).
5. Compute per-assay Spearman, Pearson, MAE on test.

Then aggregate to **macro-Spearman** (mean across assays). That's the headline
number the ProteinGym paper reports. We also dump a per-assay CSV for drilldown.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error

from .benchmark import BenchmarkRow, _embedder, append_results
from .data_proteingym import AssaySplits, load_proteingym_assays
from .heads import HEADS

logger = logging.getLogger(__name__)


def _embed_assay(emb, assay: AssaySplits, cache_dir: Path, cache_key_prefix: str
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (X_train, X_val, X_test, seconds_per_seq) for one assay (cached)."""
    cache_key = f"{cache_key_prefix}__{assay.assay_id}"
    path = cache_dir / f"{cache_key}.npy"
    meta = cache_dir / f"{cache_key}.json"
    n_train, n_val, n_test = len(assay.train), len(assay.val), len(assay.test)

    if path.exists() and meta.exists():
        logger.info("Loading cached embeddings from %s", path)
        X = np.load(path)
        info = json.loads(meta.read_text())
        if X.shape[0] != n_train + n_val + n_test:
            logger.warning("Cache shape mismatch for %s; re-embedding.", assay.assay_id)
        else:
            per_seq = float(info.get("seconds_per_seq", 0.0))
            return X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:], per_seq

    seqs = (assay.train["sequence"].tolist()
            + assay.val["sequence"].tolist()
            + assay.test["sequence"].tolist())
    X, per_seq = emb.embed(seqs, desc=f"pg/{assay.assay_id}")
    np.save(path, X)
    meta.write_text(json.dumps({"seconds_per_seq": per_seq, "shape": list(X.shape)}))
    return X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:], per_seq


def run_proteingym(
    encoder: str,
    head: str = "ridge",
    encoder_model: str | None = None,
    assays: list[str] | None = None,
    n_assays: int | None = 5,
    data_dir: str = "data/proteingym",
    cache_dir: str | Path = "cache/embeddings",
    results_dir: str | Path = "results",
) -> BenchmarkRow:
    """One (encoder, head) sweep across the chosen ProteinGym assays."""
    if head not in HEADS:
        raise ValueError(f"Unknown head {head!r}. Choose from {list(HEADS)}")

    assay_splits = load_proteingym_assays(
        assays=assays, n_assays=n_assays, data_dir=data_dir,
    )
    if not assay_splits:
        raise RuntimeError("No assays selected.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    emb = _embedder(encoder, encoder_model)
    model_name = getattr(emb, "model_name")
    cache_key_prefix = f"proteingym__{model_name.replace('/', '_')}"

    per_assay_rows: list[dict] = []
    fit_total = 0.0
    per_seq_times: list[float] = []
    n_train_total = n_val_total = n_test_total = 0
    embed_dim = None

    for assay in assay_splits:
        try:
            X_tr, X_va, X_te, per_seq = _embed_assay(
                emb, assay, cache_dir=cache_dir, cache_key_prefix=cache_key_prefix,
            )
        except Exception as e:
            logger.exception("Embedding failed for %s: %s", assay.assay_id, e)
            continue

        y_tr = assay.train["target"].to_numpy()
        y_va = assay.val["target"].to_numpy()
        y_te = assay.test["target"].to_numpy()

        t0 = time.perf_counter()
        h = HEADS[head]().fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        fit_seconds = time.perf_counter() - t0
        preds = h.predict(X_te)

        sp = spearmanr(preds, y_te).correlation
        pe = pearsonr(preds, y_te).statistic if len(y_te) > 1 else float("nan")
        mae = mean_absolute_error(y_te, preds)

        per_assay_rows.append({
            "assay_id": assay.assay_id,
            "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
            "spearman": float(sp) if sp is not None else float("nan"),
            "pearson": float(pe) if pe is not None else float("nan"),
            "mae": float(mae),
            "fit_seconds": float(fit_seconds),
        })
        per_seq_times.append(per_seq)
        fit_total += fit_seconds
        n_train_total += len(X_tr); n_val_total += len(X_va); n_test_total += len(X_te)
        if embed_dim is None:
            embed_dim = int(X_tr.shape[1])
        logger.info("[%s | %s + %s] Spearman=%.3f Pearson=%.3f MAE=%.3f",
                    assay.assay_id, encoder, head, sp or float("nan"),
                    pe if pe is not None else float("nan"), mae)

    if not per_assay_rows:
        raise RuntimeError("All assays failed.")

    per_assay_df = pd.DataFrame(per_assay_rows)
    macro_spearman = float(np.nanmean(per_assay_df["spearman"]))
    macro_pearson = float(np.nanmean(per_assay_df["pearson"]))
    macro_mae = float(np.nanmean(per_assay_df["mae"]))

    # Persist per-assay drilldown.
    date = _dt.date.today().isoformat()
    detail_path = results_dir / f"proteingym-per-assay-{encoder}-{head}-{date}.csv"
    per_assay_df.to_csv(detail_path, index=False)
    logger.info("Wrote per-assay detail to %s", detail_path)

    row = BenchmarkRow(
        date=date,
        task=f"proteingym_dms_substitutions_n{len(per_assay_df)}",
        encoder=encoder,
        encoder_model=model_name,
        head=head,
        n_train=n_train_total,
        n_val=n_val_total,
        n_test=n_test_total,
        embed_dim=int(embed_dim or 0),
        spearman=macro_spearman,
        pearson=macro_pearson,
        mae=macro_mae,
        embed_seconds_per_seq=float(np.mean(per_seq_times)) if per_seq_times else 0.0,
        fit_seconds=float(fit_total),
        extra={
            "n_assays": len(per_assay_df),
            "macro_metric": "mean_per_assay",
            "per_assay_csv": str(detail_path),
        },
    )
    logger.info("[ProteinGym | %s + %s] macro-Spearman=%.3f over %d assays",
                encoder, head, macro_spearman, len(per_assay_df))
    append_results(row, results_dir=results_dir)
    return row


def run_proteingym_all(
    encoders: tuple[str, ...] = ("esm2", "e1"),
    heads: tuple[str, ...] = ("ridge", "lasso", "mlp"),
    assays: list[str] | None = None,
    n_assays: int | None = 5,
    data_dir: str = "data/proteingym",
    cache_dir: str = "cache/embeddings",
    results_dir: str = "results",
) -> list[BenchmarkRow]:
    rows = []
    for enc in encoders:
        for head in heads:
            rows.append(run_proteingym(
                encoder=enc, head=head, assays=assays, n_assays=n_assays,
                data_dir=data_dir, cache_dir=cache_dir, results_dir=results_dir,
            ))
    return rows
