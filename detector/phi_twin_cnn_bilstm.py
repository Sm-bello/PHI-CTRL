# -*- coding: utf-8 -*-
"""
PHI-Twin detector: CNN-BiLSTM over short telemetry windows → γ̂ + fault flag.

Physics path (MMAE) remains available; this module is the *learned* health head.
Interface mirrors ElevEffectivenessBank so the orchestrator can swap/fuse.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

FEATURE_COLS = [
    "q_dps",
    "theta_deg",
    "hdot_fps",
    "vc_kts",
    "alpha_deg",
    "elevator_cmd",
    "elevator_pos",
    "throttle",
]

DEFAULT_WINDOW = 40  # samples @ ~20 Hz ≈ 2 s


class PhiTwinCNNLSTM:
    """
    Online buffer + torch model inference.
    If checkpoint missing / torch fails, .available is False (caller uses MMAE).
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        window: int = DEFAULT_WINDOW,
        device: str = "cpu",
    ):
        self.window = int(window)
        self.device = device
        self._buf: List[np.ndarray] = []
        self.model = None
        self.available = False
        self.feature_mean = None
        self.feature_std = None
        self._ckpt = Path(checkpoint) if checkpoint else None
        if self._ckpt and self._ckpt.exists():
            self._load(self._ckpt)

    def _load(self, path: Path):
        try:
            import torch
            from detector.phi_twin_model import CNNLSTMGamma
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
            self.feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
            self.window = int(ckpt.get("window", self.window))
            n_feat = len(FEATURE_COLS)
            self.model = CNNLSTMGamma(n_feat=n_feat)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            print(f"[PHI-Twin] Loaded CNN-BiLSTM from {path}")
        except Exception as e:
            print(f"[PHI-Twin] Load failed ({e}) — using physics bank only")
            self.available = False

    def reset(self):
        self._buf.clear()

    def _vector(self, sample: dict) -> np.ndarray:
        return np.array([float(sample.get(c, 0.0)) for c in FEATURE_COLS], dtype=np.float32)

    def update(self, sample: dict) -> dict:
        """
        sample must contain FEATURE_COLS keys (from flight logger / snapshot).
        Returns gamma_hat, fault_prob, confidence, source='cnn_bilstm'|'warmup'
        """
        v = self._vector(sample)
        self._buf.append(v)
        if len(self._buf) > self.window:
            self._buf = self._buf[-self.window :]

        if not self.available or len(self._buf) < self.window:
            return {
                "gamma_hat": 1.0,
                "fault_prob": 0.0,
                "confidence": 0.0,
                "source": "warmup",
            }

        import torch
        x = np.stack(self._buf, axis=0)  # (T, F)
        x = (x - self.feature_mean) / np.maximum(self.feature_std, 1e-6)
        xt = torch.from_numpy(x[None, ...]).to(self.device)  # (1, T, F)
        with torch.no_grad():
            gamma, fault_logit = self.model(xt)
            g = float(torch.clamp(gamma, 0.05, 1.0).item())
            fp = float(torch.sigmoid(fault_logit).item())
        conf = float(np.clip(abs(fp - 0.5) * 2.0, 0.0, 1.0))
        return {
            "gamma_hat": g,
            "fault_prob": fp,
            "confidence": conf,
            "source": "cnn_bilstm",
        }

    def residual_enable(self, gamma_hat: float, threshold: float = 0.92) -> bool:
        return bool(gamma_hat < threshold)
