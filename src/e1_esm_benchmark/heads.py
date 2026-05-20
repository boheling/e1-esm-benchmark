"""Downstream regression heads: Ridge, Lasso, MLP.

All three expose the same ``fit(X_train, y_train, X_val=None, y_val=None)``
/ ``predict(X)`` interface so the benchmark harness can swap them freely.

Ridge and Lasso tune alpha via grid search on the val split (or internal CV
if no val is provided). The MLP is a 2-layer PyTorch net with early stopping
on the val split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso, Ridge

logger = logging.getLogger(__name__)


@dataclass
class RidgeHead:
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    best_alpha: float = field(default=1.0, init=False)
    model: Ridge = field(init=False)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None or len(X_val) == 0:
            from sklearn.model_selection import KFold
            from sklearn.linear_model import RidgeCV
            self.model = RidgeCV(alphas=self.alphas, cv=KFold(5, shuffle=True, random_state=0))
            self.model.fit(X_train, y_train)
            self.best_alpha = float(getattr(self.model, "alpha_", 1.0))
        else:
            best = None
            best_score = -np.inf
            for a in self.alphas:
                m = Ridge(alpha=a)
                m.fit(X_train, y_train)
                s = m.score(X_val, y_val)
                if s > best_score:
                    best, best_score, self.best_alpha = m, s, a
            self.model = best
        logger.info("Ridge best alpha=%g", self.best_alpha)
        return self

    def predict(self, X):
        return self.model.predict(X)


@dataclass
class LassoHead:
    alphas: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
    best_alpha: float = field(default=0.01, init=False)
    model: Lasso = field(init=False)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None or len(X_val) == 0:
            from sklearn.linear_model import LassoCV
            self.model = LassoCV(alphas=self.alphas, cv=5, max_iter=10_000, random_state=0)
            self.model.fit(X_train, y_train)
            self.best_alpha = float(self.model.alpha_)
        else:
            best = None
            best_score = -np.inf
            for a in self.alphas:
                m = Lasso(alpha=a, max_iter=10_000, random_state=0)
                m.fit(X_train, y_train)
                s = m.score(X_val, y_val)
                if s > best_score:
                    best, best_score, self.best_alpha = m, s, a
            self.model = best
        logger.info("Lasso best alpha=%g", self.best_alpha)
        return self

    def predict(self, X):
        return self.model.predict(X)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class MLPHead:
    hidden: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _y_mean: float = field(default=0.0, init=False)
    _y_std: float = field(default=1.0, init=False)
    model: _MLP = field(init=False)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Standardize target for training stability; invert on predict.
        self._y_mean = float(np.mean(y_train))
        self._y_std = float(np.std(y_train) + 1e-8)
        yt = (y_train - self._y_mean) / self._y_std

        self.model = _MLP(X_train.shape[1], self.hidden, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(X_train).float().to(self.device)
        yt_t = torch.from_numpy(yt).float().to(self.device)

        use_val = X_val is not None and y_val is not None and len(X_val) > 0
        if use_val:
            yv = (np.asarray(y_val) - self._y_mean) / self._y_std
            Xv = torch.from_numpy(X_val).float().to(self.device)
            yv_t = torch.from_numpy(yv).float().to(self.device)

        best_val = np.inf
        bad_epochs = 0
        best_state = None

        n = len(Xt)
        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad(set_to_none=True)
                pred = self.model(Xt[idx])
                loss = loss_fn(pred, yt_t[idx])
                loss.backward()
                opt.step()
                epoch_loss += float(loss) * len(idx)
            epoch_loss /= n

            if use_val:
                self.model.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self.model(Xv), yv_t))
                if vloss < best_val - 1e-5:
                    best_val = vloss
                    bad_epochs = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    bad_epochs += 1
                if bad_epochs >= self.patience:
                    logger.info("MLP early stop at epoch %d (best val %.4f)", epoch, best_val)
                    break
            if epoch % 20 == 0:
                logger.info("MLP epoch %d train=%.4f%s", epoch, epoch_loss,
                            f" val={best_val:.4f}" if use_val else "")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.from_numpy(X).float().to(self.device)).cpu().numpy()
        return pred * self._y_std + self._y_mean


HEADS = {"ridge": RidgeHead, "lasso": LassoHead, "mlp": MLPHead}
