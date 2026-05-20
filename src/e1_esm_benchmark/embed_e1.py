"""E1 embedding extraction.

Profluent's E1 is architecturally ESM-like but does NOT use HuggingFace's
AutoModel / AutoTokenizer. Loading goes through ``E1ForMaskedLM`` plus a
custom ``E1BatchPreparer``. We mean-pool the per-residue embeddings
returned in ``outputs.embeddings``, excluding boundary tokens via
``prep.get_boundary_token_mask(input_ids)``.

Long sequences are handled with non-overlapping chunks, weighted-averaged
by the number of residues each chunk contributes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Profluent-Bio/E1-600m"
# E1's positional capacity — we conservatively chunk at 1022 to match the
# ESM2 ceiling so both models see the same amount of context per residue.
# Bump this up if you confirm E1 handles longer sequences on your hardware.
E1_MAX_RESIDUES = 1022


@dataclass
class E1Embedder:
    model_name: str = DEFAULT_MODEL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16: bool = True

    def __post_init__(self) -> None:
        # Import here so the package is usable on systems without E1 installed
        # (e.g. running only the ESM2 half of the benchmark).
        try:
            from E1.batch_preparer import E1BatchPreparer
            from E1.modeling import E1ForMaskedLM
        except ImportError as e:
            raise ImportError(
                "E1 is not installed. `pip install 'E1 @ git+https://github.com/Profluent-AI/E1.git'` "
                "or install this package with its pyproject.toml pin."
            ) from e

        logger.info("Loading %s on %s (bf16=%s)", self.model_name, self.device, self.use_bf16)
        self.model = E1ForMaskedLM.from_pretrained(self.model_name).to(self.device).eval()
        self.prep = E1BatchPreparer()

        # Infer embedding dim from a tiny probe — avoids hard-coding per model size.
        with torch.no_grad():
            probe_batch = self.prep.get_batch_kwargs(["MKA"], device=self.device)
            autocast_ctx = (
                torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16)
                if self.use_bf16 and self.device.startswith("cuda")
                else torch.cuda.amp.autocast(enabled=False) if self.device.startswith("cuda") else _nullctx()
            )
            with autocast_ctx:
                out = self.model(**probe_batch)
            self.embedding_dim = int(out.embeddings.shape[-1])
        logger.info("E1 embedding dim = %d", self.embedding_dim)

    def _pool_one(self, sequence: str) -> np.ndarray:
        """Mean-pool residue embeddings for one sequence, chunking if long."""
        chunks = [
            sequence[i : i + E1_MAX_RESIDUES]
            for i in range(0, len(sequence), E1_MAX_RESIDUES)
        ] or [""]

        total = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = 0

        for chunk in chunks:
            batch = self.prep.get_batch_kwargs([chunk], device=self.device)
            # Residue selector: NOT a boundary / pad / special token.
            residue_mask = ~self.prep.get_boundary_token_mask(batch["input_ids"])  # (B, L)

            autocast_ctx = (
                torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16)
                if self.use_bf16 and self.device.startswith("cuda")
                else _nullctx()
            )
            with torch.no_grad(), autocast_ctx:
                out = self.model(**batch)
            embs = out.embeddings[0].float().cpu().numpy()  # (L, E)
            mask = residue_mask[0].cpu().numpy()            # (L,)

            n = int(mask.sum())
            if n == 0:
                continue
            total += embs[mask].sum(axis=0)
            total_weight += n

        if total_weight == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return total / total_weight

    def embed(self, sequences: list[str], desc: str = "E1 embed") -> tuple[np.ndarray, float]:
        """Return ``(N, E)`` embeddings and mean wall-clock time per sequence (s)."""
        out = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)
        t0 = time.perf_counter()
        for i, seq in enumerate(tqdm(sequences, desc=desc)):
            out[i] = self._pool_one(seq)
        elapsed = time.perf_counter() - t0
        per_seq = elapsed / max(1, len(sequences))
        logger.info("E1 embedded %d seqs in %.1fs (%.1f ms/seq)", len(sequences), elapsed, per_seq * 1000)
        return out, per_seq


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
