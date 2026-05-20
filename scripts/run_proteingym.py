"""ProteinGym DMS-substitutions benchmark CLI.

ProteinGym is a suite of 217 deep mutational scanning assays. Each (encoder,
head) run loops over the chosen assays, fits a head per assay, and reports
**macro-Spearman** = mean per-assay Spearman.

Examples
--------
    # Quick test: 5 assays, both encoders x all 3 heads
    python scripts/run_proteingym.py --all

    # Single encoder/head, 20 assays
    python scripts/run_proteingym.py --encoder e1 --head ridge --n-assays 20

    # Specific assays (comma-separated DMS_id values)
    python scripts/run_proteingym.py --all --assays BLAT_ECOLX_Stiffler_2015,GFP_AEQVI_Sarkisyan_2016

    # Full 217-assay sweep (slow!)
    python scripts/run_proteingym.py --all --n-assays 0    # 0 means "all"
"""

from __future__ import annotations

import argparse
import logging

from e1_esm_benchmark.benchmark import ENCODERS
from e1_esm_benchmark.benchmark_proteingym import run_proteingym, run_proteingym_all


def main() -> int:
    p = argparse.ArgumentParser(description="ProteinGym DMS-substitutions benchmark: E1 vs ESM2.")
    p.add_argument("--encoder", choices=ENCODERS, default="esm2")
    p.add_argument("--head", choices=("ridge", "lasso", "mlp"), default="ridge")
    p.add_argument("--encoder-model", default=None)
    p.add_argument("--assays", default=None,
                   help="Comma-separated DMS_id values. Overrides --n-assays.")
    p.add_argument("--n-assays", type=int, default=5,
                   help="Number of assays to run when --assays is not set. 0 = all 217.")
    p.add_argument("--data-dir", default="data/proteingym")
    p.add_argument("--cache-dir", default="cache/embeddings")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--all", action="store_true",
                   help="Sweep both encoders x all 3 heads.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    assay_list = [a.strip() for a in args.assays.split(",")] if args.assays else None
    n_assays = None if args.n_assays == 0 else args.n_assays

    if args.all:
        run_proteingym_all(
            assays=assay_list, n_assays=n_assays,
            data_dir=args.data_dir, cache_dir=args.cache_dir,
            results_dir=args.results_dir,
        )
    else:
        run_proteingym(
            encoder=args.encoder, head=args.head, encoder_model=args.encoder_model,
            assays=assay_list, n_assays=n_assays,
            data_dir=args.data_dir, cache_dir=args.cache_dir,
            results_dir=args.results_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
