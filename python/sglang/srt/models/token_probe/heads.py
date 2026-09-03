from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, ReplicatedLinear
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.token_probe.config import ProbeConfig
from sglang.srt.models.token_probe.probe_kernels import (
    ACT_GELU,
    ACT_RELU,
    classify_tail,
    fused_add_rmsnorm_classifier_sigmoid,
)


def _validate_layer_ids(layer_ids: tuple[int, ...]) -> None:
    if not layer_ids:
        raise ValueError("token probe config must list 'base_model_layer_ids'")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("token probe base_model_layer_ids must be unique")


class IdentityProbeHead(nn.Module):
    """Return normalized, concatenated tapped states without trainable weights."""

    def __init__(
        self,
        base_model_layer_ids: tuple[int, ...],
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        _validate_layer_ids(base_model_layer_ids)
        self._base_model_layer_ids = base_model_layer_ids
        self._hidden_size = hidden_size

    @classmethod
    def from_config(
        cls, config: ProbeConfig, dtype: Optional[torch.dtype] = None
    ) -> "IdentityProbeHead":
        return cls(
            base_model_layer_ids=config.base_model_layer_ids,
            hidden_size=config.hidden_size,
        )

    @property
    def hidden_size(self) -> int | None:
        return self._hidden_size

    @property
    def state_indices(self) -> tuple[int, ...]:
        return self._base_model_layer_ids

    @property
    def label_names(self) -> tuple:
        return ()

    def forward_features(self, features: torch.Tensor, aggregate: bool = True) -> dict:
        return {"probe_score": features}

    def load_weights(self, weights) -> None:
        raise ValueError("the identity token probe takes no weights")


def _activation_code(name: str) -> int:
    if name == "gelu":
        return ACT_GELU
    if name == "relu":
        return ACT_RELU
    raise ValueError(f"Unsupported activation {name!r}; use 'gelu' or 'relu'.")


class SingProbeMlpModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        num_labels: int,
        base_model_layer_ids: tuple[int, ...],
        hidden_size: int,
        hidden_act: str = "gelu",
        params_dtype: Optional[torch.dtype] = None,
        labels: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        _validate_layer_ids(base_model_layer_ids)
        self._base_model_layer_ids = base_model_layer_ids
        self._hidden_size = hidden_size
        self._labels = (
            tuple(labels)
            if labels
            else tuple(f"label_{index}" for index in range(num_labels))
        )
        self.fc1 = ReplicatedLinear(
            input_size,
            intermediate_size,
            params_dtype=params_dtype,
        )
        self._act = _activation_code(hidden_act)
        self.fc2 = ReplicatedLinear(
            intermediate_size,
            num_labels,
            params_dtype=params_dtype,
        )

    @classmethod
    def from_config(
        cls, config: ProbeConfig, dtype: Optional[torch.dtype] = None
    ) -> "SingProbeMlpModel":
        if config.input_size is None:
            raise ValueError("MLP token probe requires hidden_size")
        return cls(
            input_size=config.input_size,
            intermediate_size=config.intermediate_size,
            num_labels=config.num_labels,
            base_model_layer_ids=config.base_model_layer_ids,
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            params_dtype=dtype,
            labels=config.labels,
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def state_indices(self) -> tuple[int, ...]:
        return self._base_model_layer_ids

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._labels

    @property
    def num_labels(self) -> int:
        return len(self._labels)

    def forward_features(self, features: torch.Tensor, aggregate: bool = True) -> dict:
        hidden, _ = self.fc1(features.to(self.fc1.weight.dtype))
        return {
            "probe_score": classify_tail(
                hidden, self.fc2.weight, self.fc2.bias, self._act
            )
        }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded = set()
        unexpected = set()
        for name, weight in weights:
            param = params.get(name)
            if param is None:
                unexpected.add(name)
                continue
            param.data.copy_(weight.to(param.dtype))
            loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"SingProbe MLP head is missing weights {sorted(missing)}")
        if unexpected:
            raise ValueError(
                f"SingProbe MLP head has unexpected weights {sorted(unexpected)}"
            )


class SingProbeAttnModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        base_model_layer_ids: tuple[int, ...],
        num_attention_heads: int = 4,
        head_dim: int = 64,
        sliding_window: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        labels: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        _validate_layer_ids(base_model_layer_ids)
        if num_attention_heads < 1 or head_dim < 1:
            raise ValueError("num_attention_heads and head_dim must be positive")
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("token probe sliding_window must be positive")
        self._sliding_window = sliding_window
        self._hidden_size = hidden_size
        self._base_model_layer_ids = base_model_layer_ids
        self._labels = (
            tuple(labels)
            if labels
            else tuple(f"label_{index}" for index in range(num_labels))
        )
        input_size = len(base_model_layer_ids) * hidden_size
        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_kv_heads = 1
        self._projection_size = num_attention_heads * head_dim

        # Keep one fused projection at runtime while accepting the canonical
        # SingProbe checkpoint's separate proj_q/proj_k/proj_v weights.
        self.proj_qkv = QKVParallelLinear(
            hidden_size=input_size,
            head_size=head_dim,
            total_num_heads=num_attention_heads,
            total_num_kv_heads=1,
            bias=False,
            params_dtype=params_dtype,
            tp_rank=0,
            tp_size=1,
            prefix="proj_qkv",
        )
        self.o_proj = ReplicatedLinear(
            self._projection_size,
            self._projection_size,
            bias=False,
            params_dtype=params_dtype,
            prefix="o_proj",
        )
        self.norm = RMSNorm(
            self._projection_size, eps=1e-6, weight_dtype=params_dtype
        )
        self.classifier = ReplicatedLinear(
            self._projection_size,
            num_labels,
            bias=True,
            params_dtype=params_dtype,
            prefix="classifier",
        )

    @classmethod
    def from_config(
        cls, config: ProbeConfig, dtype: Optional[torch.dtype] = None
    ) -> "SingProbeAttnModel":
        if config.hidden_size is None:
            raise ValueError("attention token probe requires hidden_size")
        return cls(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            base_model_layer_ids=config.base_model_layer_ids,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            sliding_window=config.sliding_window,
            params_dtype=dtype,
            labels=config.labels,
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def state_indices(self) -> tuple[int, ...]:
        return self._base_model_layer_ids

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._labels

    @property
    def num_labels(self) -> int:
        return len(self._labels)

    @property
    def sliding_window(self) -> int | None:
        return self._sliding_window

    @property
    def q_dim(self) -> int:
        return self._projection_size

    @property
    def kv_dim(self) -> int:
        return self._num_kv_heads * self._head_dim

    @property
    def num_attention_heads(self) -> int:
        return self._num_attention_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def kv_dtype(self) -> torch.dtype:
        return self.proj_qkv.weight.dtype

    def project_fused(self, features: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.proj_qkv(features.to(self.proj_qkv.weight.dtype))
        return qkv

    def classify(self, attention_output: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        projected, _ = self.o_proj(attention_output.to(self.o_proj.weight.dtype))
        return fused_add_rmsnorm_classifier_sigmoid(
            projected,
            query,
            self.norm.weight,
            self.classifier.weight,
            self.classifier.bias,
            self.norm.variance_epsilon,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        qkv_shards = {
            "proj_q.weight": "q",
            "proj_k.weight": "k",
            "proj_v.weight": "v",
        }
        params = dict(self.named_parameters())
        expected = (set(params) - {"proj_qkv.weight"}) | set(qkv_shards)
        loaded = set()
        unexpected = set()
        for name, weight in weights:
            if name.startswith("attn_layer."):
                raise ValueError(
                    "This checkpoint contains the removed cross-layer attention "
                    "pooling weights; retrain or re-export the SingProbe head."
                )
            shard = qkv_shards.get(name)
            if shard is not None:
                param = params["proj_qkv.weight"]
                param.weight_loader(param, weight, shard)
            else:
                param = params.get(name)
                if param is None:
                    unexpected.add(name)
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)
            loaded.add(name)
        missing = expected - loaded
        if missing:
            raise ValueError(
                f"SingProbe attention head is missing weights {sorted(missing)}"
            )
        if unexpected:
            raise ValueError(
                f"SingProbe attention head has unexpected weights {sorted(unexpected)}"
            )
