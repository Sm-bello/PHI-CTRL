# -*- coding: utf-8 -*-
"""CNN-BiLSTM backbone for PHI-Twin γ̂ + fault classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTMGamma(nn.Module):
    """
    Input:  (B, T, F) window of telemetry
    Output: gamma in (0,1) via sigmoid, fault logit
    """

    def __init__(self, n_feat: int = 8, cnn_channels: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        hid = 2 * lstm_hidden
        self.head_gamma = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.head_fault = nn.Linear(hid, 1)

    def forward(self, x):
        # x: (B, T, F) → conv wants (B, F, T)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        out, _ = self.lstm(z)
        h = out[:, -1, :]  # last timestep
        gamma = self.head_gamma(h).squeeze(-1)
        fault_logit = self.head_fault(h).squeeze(-1)
        return gamma, fault_logit
