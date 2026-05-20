"""ESM2 embedding extraction.

We take the final hidden state of ``facebook/esm2_t33_650M_UR50D`` and
mean-pool over residue tokens only (excluding ``<cls>``, ``<eos>``, ``<pad>``).
Sequences longer than ``max_length - 2`` residues are handled with a
non-overlapping sliding window; per-window mean embeddings are averaged
weighted by the number of residues each window contributed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
# ESM2 was trained with 1024 positions total (including special tokens),
# so 1022 residues per chunk is the safe ceiling.
ESM_MAX_RESIDUES = 1022


@dataclass
class EsmEmbedder:
    model_name: str = DEFAULT_MODEL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        logger.info("Loading %s on %s (%s)", self.model_name, self.device, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.model.eval().to(self.device)
        self.embedding_dim = self.model.config.hidden_size
        # IDs we want to exclude from the pool.
        self._special_ids = {
            i
            for i in [
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.bos_token_id,
            ]
            if i is not None
        }

    def _pool_one(self, sequence: str) -> np.ndarray:
        """Mean-pool residue embeddings for a single sequence (with chunking)."""
        chunks = [
            sequence[i : i + ESM_MAX_RESIDUES]
            for i in range(0, len(sequence), ESM_MAX_RESIDUES)
        ] or [""]

        total = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = 0

        for chunk in chunks:
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=ESM_MAX_RESIDUES + 2,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state[0].float().cpu().numpy()  # (L, E)
            ids = input_ids[0].cpu().numpy()

            # Residue mask: attended AND not a special token.
            resid_mask = (attention_mask[0].cpu().numpy() == 1) & ~np.isin(
                ids, np.array(list(self._special_ids), dtype=ids.dtype)
            )
            n = int(resid_mask.sum())
            if n == 0:
                continue
            total += hidden[resid_mask].sum(axis=0)
            total_weight += n

        if total_weight == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return total / total_weight

    def embed(self, sequences: list[str], desc: str = "ESM2 embed") -> tuple[np.ndarray, float]:
        """Return ``(N, E)`` embeddings and the mean wall-clock time per sequence (s)."""
        out = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)
        t0 = time.perf_counter()
        for i, seq in enumerate(tqdm(sequences, desc=desc)):
            out[i] = self._pool_one(seq)
        elapsed = time.perf_counter() - t0
        per_seq = elapsed / max(1, len(sequences))
        logger.info("ESM2 embedded %d seqs in %.1fs (%.1f ms/seq)", len(sequences), elapsed, per_seq * 1000)
        return out, per_seq
