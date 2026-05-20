# e1-esm-benchmark

**Is Profluent's E1 actually a drop-in replacement for ESM2?**

E1 is [pitched](https://github.com/Profluent-AI/E1) as a drop-in replacement for the ESM family. In the HuggingFace `AutoModel` sense it isn't — E1 ships its own loader and batch preparer, and there's no `E1ForSequenceClassification` head. But in the more interesting sense — *do the frozen embeddings carry comparable signal for downstream regression?* — the question is empirical.

This repo runs that comparison on a public benchmark. Frozen encoder, sparse head, same pooling, same splits, same metrics.

## TL;DR

Four tasks span thermostability, binding fitness, functional fluorescence, and variant effect:

| Task            | Source                          | Type                                | Sequences       |
|-----------------|---------------------------------|-------------------------------------|-----------------|
| `meltome`       | FLIP Meltome (mixed split)      | Thermostability regression (°C)     | ~33k, ≤2k aa    |
| `gb1`           | FLIP GB1 (sampled split)        | Binding-fitness regression          | ~150k, ~56 aa   |
| `fluorescence`  | TAPE / proteinea fluorescence   | Log-fluorescence regression (GFP)   | ~54k, ~236 aa   |
| `proteingym`    | ProteinGym v1 DMS substitutions | Macro-Spearman over DMS assays      | suite of 217    |

| Task         | Encoder    | Head  | Spearman | Pearson | MAE  | Embed dim |
|--------------|------------|-------|---------:|--------:|-----:|----------:|
| meltome      | ESM2-650M  | Ridge |      TBD |     TBD |  TBD |      1280 |
| meltome      | E1-600m    | Ridge |      TBD |     TBD |  TBD |       TBD |
| gb1          | ESM2-650M  | Ridge |      TBD |     TBD |  TBD |      1280 |
| gb1          | E1-600m    | Ridge |      TBD |     TBD |  TBD |       TBD |
| fluorescence | ESM2-650M  | Ridge |      TBD |     TBD |  TBD |      1280 |
| fluorescence | E1-600m    | Ridge |      TBD |     TBD |  TBD |       TBD |
| proteingym   | ESM2-650M  | Ridge |      TBD |     TBD |  TBD |      1280 |
| proteingym   | E1-600m    | Ridge |      TBD |     TBD |  TBD |       TBD |

_Numbers will land here after the first full sweep. See `results/comparison-<task>-YYYY-MM-DD.md` for the latest auto-generated tables, and `results/proteingym-per-assay-*.csv` for per-assay drilldown._

## Quick start

```bash
git clone https://github.com/boheling/e1-esm-benchmark
cd e1-esm-benchmark
pip install -e .

# Sweep one task (both encoders x ridge/lasso/mlp). Default task is meltome.
python scripts/run_benchmark.py --task meltome --all
python scripts/run_benchmark.py --task gb1 --all
python scripts/run_benchmark.py --task fluorescence --all

# ProteinGym is a *suite* — it has its own runner that aggregates across assays.
python scripts/run_proteingym.py --all                     # 5 assays (default)
python scripts/run_proteingym.py --all --n-assays 20       # broader sweep
python scripts/run_proteingym.py --all --n-assays 0        # all 217 (slow)
```

Run time on a single A100: ≈ 1–2 h for Meltome, ≈ 1 h for GB1, ≈ 30 min for fluorescence, and roughly 20 min × n_assays for ProteinGym. Most of it is one-shot embedding extraction, cached under `cache/embeddings/` after the first run.

### No local GPU? Use Kaggle or Colab

Open [`notebooks/run_benchmark.ipynb`](notebooks/run_benchmark.ipynb) on **Kaggle** (preferred — 12 h sessions, ~30 GPU-hours/week free on T4 or P100) or **Google Colab** (T4, idle-disconnects ~90 min). The notebook auto-detects the environment, mounts persistent storage (Kaggle Datasets / Google Drive) so the embedding cache survives across sessions, installs the repo, and runs the chosen task. T4 is ~3× slower than A100 — `fluorescence` is the fastest sweep to try first.

## Tasks

**`meltome` — FLIP Meltome (mixed split).** Thermostability regression over ~33k proteins pooled from 13 organisms ([Jarzab et al. 2020](https://www.nature.com/articles/s41592-020-0801-4), curated into FLIP by [Dallago et al. 2021](https://openreview.net/forum?id=p2dMLEwL8tF)). Target is melting temperature (°C). We use the mixed split for a smoother comparison.

**`gb1` — FLIP GB1 (sampled split).** Binding-fitness regression over ~150k variants of the GB1 immunoglobulin-binding domain (Wu et al. 2016; FLIP). Sequences are ~56 aa; the four hotspot positions vary combinatorially. No length-chunking needed — this isolates dense-local representation quality.

**`fluorescence` — TAPE Fluorescence.** Log-fluorescence regression over ~54k GFP variants (Sarkisyan et al. 2016; TAPE benchmark, mirrored as `proteinea/fluorescence` on HF). 0–3 point mutations from the avGFP wildtype. Short, clean, well-baselined.

**`proteingym` — ProteinGym v1 DMS substitutions.** Macro-Spearman across ~217 deep-mutational-scan assays ([Notin et al. 2023](https://proteingym.org/)). This is *E1's home turf* — variant effect prediction is one of its named pre-training targets, so it's the most adversarial test of "drop-in replacement." Per-assay 80/10/10 random split, ridge/lasso/MLP head, average Spearman across assays.

## Design choices

**Frozen encoder + sparse head.** Both encoders are run inference-only with no fine-tuning. We mean-pool per-residue embeddings excluding boundary/pad tokens, then fit a Ridge / Lasso / MLP head. This isolates the quality of the representation from head capacity and LoRA hyperparameters, and echoes a result from a prior benchmark (HLA epitope recovery): a sparse Lasso on engineered features beat a 650M-parameter protein LM at the biological task, suggesting representation quality matters more than head capacity when the signal is structured.

**Matched preprocessing.** Same residue sanitization, same length cap, same splits for both models. The only difference is the encoder.

**Long sequences.** Both embedders chunk at 1022 residues (ESM2's max minus special tokens) with mean-of-chunk pooling weighted by residue count. We keep the ceiling identical for both models so long-sequence handling doesn't leak a hidden variable.

**Cached embeddings.** First run writes `cache/embeddings/{model}__{split}.npy`. Subsequent head-only sweeps are seconds, not hours.

## Repo layout

```
e1-esm-benchmark/
├── pyproject.toml                   # deps incl. E1 @ git+https://github.com/Profluent-AI/E1.git
├── src/e1_esm_benchmark/
│   ├── data.py                      # FLIP Meltome / GB1 / TAPE Fluorescence loaders + TASKS registry
│   ├── data_proteingym.py           # ProteinGym per-assay loader
│   ├── embed_esm.py                 # ESM2 mean-pool embeddings (HF transformers)
│   ├── embed_e1.py                  # E1 mean-pool embeddings (Profluent custom API)
│   ├── heads.py                     # Ridge / Lasso / MLP
│   ├── benchmark.py                 # single-task orchestrator + results writer
│   └── benchmark_proteingym.py      # multi-assay orchestrator (macro-Spearman)
├── scripts/
│   ├── run_benchmark.py             # CLI for meltome / gb1 / fluorescence
│   ├── run_proteingym.py            # CLI for ProteinGym multi-assay sweeps
│   └── download_data.py             # optional one-shot data warmup
├── configs/
│   ├── esm2_650m.yaml
│   └── e1_600m.yaml
└── results/                         # auto-generated CSV + per-task markdown + per-assay CSV
```

## How to extend

- **Different single-protein task.** Add a `load_<name>()` to `data.py` returning `Splits(train, val, test, task=...)` (each DataFrame: `sequence, target`) and register it in `TASKS`. Everything else routes automatically.
- **Different multi-assay task.** Add a loader returning `list[AssaySplits]` and pattern-match `benchmark_proteingym.py`.
- **Different encoder.** Add `embed_{name}.py` with a class that implements `embed(sequences) -> (np.ndarray, float)`. Register it in `benchmark._embedder()`.
- **Different head.** Add a class with `.fit(X_train, y_train, X_val, y_val)` / `.predict(X)` to `heads.py` and register it in `heads.HEADS`.
- **Fine-tune instead of frozen.** Out of scope for this repo — use the respective native training API. The point here is the frozen-feature comparison.

## Caveats

- **Task mismatch risk.** E1 is pre-trained for substitution fitness, contact maps, SSM, and zero-shot variant effect; ESM2 is trained with general masked LM. Meltome / GB1 / Fluorescence are *not* the explicit training target for either model. ProteinGym is the adversarial case — variant effect is in E1's named pre-training objectives.
- **ProteinGym defaults to a 5-assay subset.** The full 217-assay sweep is hours of GPU. The CLI lets you scale up or down. Per-assay results are dumped to a CSV so you can inspect which assays each encoder wins.
- **Boundary-token definitions differ.** ESM2 has `<cls>` / `<eos>`; E1 uses its own `get_boundary_token_mask`. We trust each model's mask rather than handcrafting one.
- **Model sizes aren't identical.** ESM2-650M vs E1-600m. Close enough for apples-to-apples but not bit-exact. The point isn't "which is bigger" — it's "for the same downstream compute, which representation wins."
- **License on E1 weights.** Apache 2.0 on code, separate terms on weights with attribution requirements. See the E1 repo's `ATTRIBUTION` file before redistributing.

## Cite

If this benchmark shaped a decision, cite it as:

> Liu, Jing. _e1-esm-benchmark: Frozen-feature comparison of Profluent E1 vs Meta ESM2 on FLIP Meltome._ GitHub, 2026.

And cite the original work:

- **E1** — [Profluent-AI/E1](https://github.com/Profluent-AI/E1)
- **ESM2** — Lin et al., "Evolutionary-scale prediction of atomic level protein structure," _Science_ 379, 2023.
- **FLIP** — Dallago et al., "FLIP: Benchmark tasks in fitness landscape inference for proteins," NeurIPS Datasets 2021.

## Author

Jing Liu — boheling@gmail.com

## License

MIT.
