"""Benchmark CLI.

Examples
--------
    # Single run on the default task (meltome):
    python scripts/run_benchmark.py --encoder esm2 --head ridge

    # Pick a different task:
    python scripts/run_benchmark.py --task gb1 --encoder e1 --head ridge
    python scripts/run_benchmark.py --task fluorescence --encoder esm2 --head mlp

    # Full sweep (both encoders x Ridge/Lasso/MLP) on one task:
    python scripts/run_benchmark.py --task meltome --all
    python scripts/run_benchmark.py --task gb1 --all
    python scripts/run_benchmark.py --task fluorescence --all

For ProteinGym (multi-assay aggregation), use ``scripts/run_proteingym.py``.
"""

from __future__ import annotations

import argparse
import logging

from e1_esm_benchmark.benchmark import ENCODERS, append_results, run_all, run_one
from e1_esm_benchmark.data import TASKS


def main() -> int:
    p = argparse.ArgumentParser(description="Frozen-encoder benchmark: E1 vs ESM2 on protein regression.")
    p.add_argument("--task", choices=list(TASKS), default="meltome",
                   help="Which benchmark task to run.")
    p.add_argument("--encoder", choices=ENCODERS, default="esm2")
    p.add_argument("--head", choices=("ridge", "lasso", "mlp"), default="ridge")
    p.add_argument("--encoder-model", default=None, help="Override the HF/Profluent model ID.")
    p.add_argument("--data-dir", default=None,
                   help="Where to cache the task's data. Defaults to data/<task>.")
    p.add_argument("--cache-dir", default="cache/embeddings")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--max-length", type=int, default=1022,
                   help="Length cap for Meltome train/val. Ignored by other tasks.")
    p.add_argument("--all", action="store_true", help="Run all (encoder x head) combos for the chosen task.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.all:
        run_all(
            task=args.task,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
        )
    else:
        row = run_one(
            task=args.task,
            encoder=args.encoder,
            head=args.head,
            encoder_model=args.encoder_model,
            data_dir=args.data_dir,
            max_length=args.max_length,
            cache_dir=args.cache_dir,
        )
        append_results(row, results_dir=args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
