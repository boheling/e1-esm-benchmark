"""One-shot FLIP Meltome download.

You usually don't need to run this directly — ``data.load_meltome()`` will
download on first access. Useful for warming the cache on a fresh checkout.
"""

from __future__ import annotations

import argparse
import logging

from e1_esm_benchmark.data import load_meltome


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/flip_meltome")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    splits = load_meltome(data_dir=args.data_dir)
    print(splits.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
