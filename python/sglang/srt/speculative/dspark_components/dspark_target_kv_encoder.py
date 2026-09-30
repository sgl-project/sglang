"""Trainable pointwise target-KV encoder with one serving/training definition."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    TargetKVDraftContract,
)
from sglang.srt.training_capture.kv_codec import target_kv_features
from torch import nn


class TargetKVContextEncoder(nn.Module):
    def __init__(self, contract: TargetKVDraftContract):
        super().__init__()
        self.contract = contract
        self.projection = nn.Linear(
            contract.feature_size, contract.encoder.hidden_size, bias=False
        )
        self.norm_weight = nn.Parameter(torch.ones(contract.encoder.hidden_size))

    def forward(
        self, tensors: Mapping[str, torch.Tensor], positions: torch.Tensor
    ) -> torch.Tensor:
        features = target_kv_features(
            self.contract.kv,
            tensors,
            positions,
            feature_k_stage=self.contract.feature_k_stage,
        )
        hidden = F.linear(
            features.to(self.projection.weight.dtype), self.projection.weight
        )
        value = hidden.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.contract.encoder.rms_norm_eps)
        return (value * self.norm_weight.float()).to(hidden.dtype)
