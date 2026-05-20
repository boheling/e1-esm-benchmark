"""Benchmark orchestrator.

For a chosen (task, encoder, head), this module:

1. Loads task splits via :data:`data.TASKS`
2. Extracts train/val/test embeddings from the encoder
3. Fits the head on train (with val for alpha tuning / early stopping)
4. Computes Spearman rho + MAE + Pearson on test
5. Appends a row to ``results/benchmark.csv`` and a per-(task, date) markdown
   table

The ``run_all`` helper sweeps (encoder x head) combinations in one process.
ProteinGym has its own orchestrator in :mod:`benchmark_proteingym` because it
aggregates over a suite of assays rather than fitting one regression.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error

from .data import TASKS
from .heads import HEADS

logger = logging.getLogger(__name__)

ENCODERS = ("esm2", "e1")
DEFAULT_TASK = "meltome"


@dataclass
class BenchmarkRow:
    date: str
    task: str
    encoder: str
    encoder_model: str
    head: str
    n_train: int
    n_val: int
    n_test: int
    embed_dim: int
    spearman: float
    pearson: float
    mae: float
    embed_seconds_per_seq: float
    fit_seconds: float
    extra: dict = field(default_factory=dict)


def _embedder(encoder: str, model_name: str | None = None):
    if encoder == "esm2":
        from .embed_esm import DEFAULT_MODEL, EsmEmbedder
        return EsmEmbedder(model_name=model_name or DEFAULT_MODEL)
    if encoder == "e1":
        from .embed_e1 import DEFAULT_MODEL, E1Embedder
        return E1Embedder(model_name=model_name or DEFAULT_MODEL)
    raise ValueError(f"Unknown encoder {encoder!r}. Choose from {ENCODERS}.")


def _load_task(task: str, **loader_kwargs):
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose from {list(TASKS)}.")
    return TASKS[task](**loader_kwargs)


def run_one(
    encoder: str,
    head: str = "ridge",
    task: str = DEFAULT_TASK,
    encoder_model: str | None = None,
    data_dir: str | None = None,
    max_length: int | None = 1022,
    cache_dir: str | Path = "cache/embeddings",
    loader_kwargs: dict | None = None,
) -> BenchmarkRow:
    """Run a single (task, encoder, head) benchmark pass."""
    loader_kwargs = dict(loader_kwargs or {})
    if data_dir is not None:
        loader_kwargs.setdefault("data_dir", data_dir)
    # max_length only applies to tasks that respect it (meltome). Others ignore.
    if task == "meltome" and "max_length" not in loader_kwargs:
        loader_kwargs["max_length"] = max_length

    splits = _load_task(task, **loader_kwargs)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    emb = _embedder(encoder, encoder_model)
    model_name = getattr(emb, "model_name")
    cache_key = f"{splits.task}__{model_name.replace('/', '_')}"

    def _cached_embed(split_name: str, seqs: list[str]) -> tuple[np.ndarray, float]:
        path = cache_dir / f"{cache_key}__{split_name}.npy"
        meta = cache_dir / f"{cache_key}__{split_name}.json"
        if path.exists() and meta.exists():
            logger.info("Loading cached embeddings from %s", path)
            X = np.load(path)
            per_seq = float(json.loads(meta.read_text()).get("seconds_per_seq", 0.0))
            return X, per_seq
        X, per_seq = emb.embed(seqs, desc=f"{encoder} {splits.task} {split_name}")
        np.save(path, X)
        meta.write_text(json.dumps({"seconds_per_seq": per_seq, "shape": list(X.shape)}))
        return X, per_seq

    X_train, t_train = _cached_embed("train", splits.train["sequence"].tolist())
    X_val, t_val = _cached_embed("val", splits.val["sequence"].tolist())
    X_test, t_test = _cached_embed("test", splits.test["sequence"].tolist())
    y_train = splits.train["target"].to_numpy()
    y_val = splits.val["target"].to_numpy()
    y_test = splits.test["target"].to_numpy()

    if head not in HEADS:
        raise ValueError(f"Unknown head {head!r}. Choose from {list(HEADS)}")

    t0 = time.perf_counter()
    h = HEADS[head]().fit(X_train, y_train, X_val=X_val, y_val=y_val)
    fit_seconds = time.perf_counter() - t0

    preds = h.predict(X_test)
    spearman = float(spearmanr(preds, y_test).correlation)
    pearson = float(pearsonr(preds, y_test).statistic)
    mae = float(mean_absolute_error(y_test, preds))

    per_seq_mean = float(np.mean([t_train, t_val, t_test]))
    row = BenchmarkRow(
        date=_dt.date.today().isoformat(),
        task=splits.task,
        encoder=encoder,
        encoder_model=model_name,
        head=head,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        embed_dim=int(X_train.shape[1]),
        spearman=spearman,
        pearson=pearson,
        mae=mae,
        embed_seconds_per_seq=per_seq_mean,
        fit_seconds=fit_seconds,
        extra={"best_alpha": getattr(h, "best_alpha", None)},
    )
    logger.info(
        "[%s | %s + %s] Spearman=%.3f  Pearson=%.3f  MAE=%.3f  embed=%.0fms/seq  fit=%.1fs",
        splits.task, encoder, head, spearman, pearson, mae, per_seq_mean * 1000, fit_seconds,
    )
    return row


def append_results(row: BenchmarkRow, results_dir: str | Path = "results") -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "benchmark.csv"
    rec = asdict(row)
    rec["extra"] = json.dumps(rec["extra"])
    df_new = pd.DataFrame([rec])
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)

    md_path = results_dir / f"comparison-{row.task}-{row.date}.md"
    same = df[(df["date"] == row.date) & (df["task"] == row.task)]
    table_cols = ["encoder", "encoder_model", "head", "spearman", "pearson", "mae",
                  "embed_dim", "embed_seconds_per_seq"]
    md = [f"# {row.task} — E1 vs ESM2 (frozen encoder + head)", "",
          f"_Run date: {row.date}. Task: `{row.task}`. Encoders: ESM2-650M, E1-600m. "
          f"Heads: Ridge / Lasso / MLP._", ""]
    md.append(same[table_cols].to_markdown(index=False, floatfmt=".3f"))
    md.append("")
    md.append("Interpretation: higher Spearman / Pearson is better; lower MAE is better. "
              "Embed time is wall-clock per sequence, single GPU.")
    md_path.write_text("\n".join(md))
    logger.info("Wrote %s and %s", csv_path, md_path)


def run_all(
    task: str = DEFAULT_TASK,
    encoders: tuple[str, ...] = ENCODERS,
    heads: tuple[str, ...] = ("ridge", "lasso", "mlp"),
    data_dir: str | None = None,
    cache_dir: str = "cache/embeddings",
    results_dir: str = "results",
    loader_kwargs: dict | None = None,
) -> list[BenchmarkRow]:
    """Sweep (encoder x head) for one task. Embeddings cached after the first pass."""
    rows: list[BenchmarkRow] = []
    for enc in encoders:
        for head in heads:
            row = run_one(
                encoder=enc, head=head, task=task,
                data_dir=data_dir, cache_dir=cache_dir,
                loader_kwargs=loader_kwargs,
            )
            append_results(row, results_dir=results_dir)
            rows.append(row)
    return rows
