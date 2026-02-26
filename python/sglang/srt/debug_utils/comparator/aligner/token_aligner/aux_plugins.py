from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    SeqId,
    SGLangSeqId,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink

_BSHD_NOT_SUPPORTED_MSG: str = (
    "BSHD layout is not currently supported. "
    "Use aux_loader BSHD→THD conversion (planned)."
)


# ── plugin ABC ─────────────────────────────────────────────────────


class _AuxFrameworkPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def tensor_names(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def non_tensor_names(self) -> frozenset[str]: ...

    @property
    def cp_sharded_names(self) -> frozenset[str]:
        return frozenset()

    @property
    def discriminating_names(self) -> frozenset[str]:
        """Field names unique to this framework (excluding shared names like input_ids)."""
        return frozenset()

    @abstractmethod
    def detect_layout(self, raw: dict[int, dict[str, object]]) -> str: ...

    @abstractmethod
    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: str, step: int
    ) -> TokenAlignerStepAux: ...

    @abstractmethod
    def has_required_names(self, names: set[str]) -> bool:
        """Whether the minimum set of aux names needed for alignment is present."""
        ...

    @property
    def all_names(self) -> frozenset[str]:
        return self.tensor_names | self.non_tensor_names


# ── sglang plugin ─────────────────────────────────────────────────


class _SGLangPlugin(_AuxFrameworkPlugin):
    @property
    def name(self) -> str:
        return "sglang"

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "positions", "seq_lens", "req_pool_indices"})

    @property
    def non_tensor_names(self) -> frozenset[str]:
        return frozenset({"rids"})

    @property
    def cp_sharded_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "positions"})

    @property
    def discriminating_names(self) -> frozenset[str]:
        return frozenset({"seq_lens", "positions", "req_pool_indices", "rids"})

    def has_required_names(self, names: set[str]) -> bool:
        return "input_ids" in names and "seq_lens" in names

    def detect_layout(self, raw: dict[int, dict[str, object]]) -> str:
        return "thd"

    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: str, step: int
    ) -> TokenAlignerStepAux:
        input_ids = step_data["input_ids"]
        positions = step_data["positions"]
        seq_lens = step_data["seq_lens"]
        rids_raw = step_data.get("rids")

        assert isinstance(
            input_ids, torch.Tensor
        ), f"input_ids: expected Tensor, got {type(input_ids)}"
        assert isinstance(
            positions, torch.Tensor
        ), f"positions: expected Tensor, got {type(positions)}"
        assert isinstance(
            seq_lens, torch.Tensor
        ), f"seq_lens: expected Tensor, got {type(seq_lens)}"

        seq_lens_list: list[int] = seq_lens.tolist()
        num_seqs: int = len(seq_lens_list)

        seq_ids: list[SeqId]
        if rids_raw is not None and isinstance(rids_raw, (list, tuple)):
            seq_ids = [SGLangSeqId(rid=str(r)) for r in rids_raw]
        else:
            seq_ids = [PositionalSeqId(step=step, seq_index=i) for i in range(num_seqs)]

        return TokenAlignerStepAux(
            input_ids=input_ids.tolist(),
            positions=positions.tolist(),
            seq_lens=seq_lens_list,
            seq_ids=seq_ids,
        )


# ── megatron plugin ───────────────────────────────────────────────


class _MegatronPlugin(_AuxFrameworkPlugin):
    @property
    def name(self) -> str:
        return "megatron"

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "position_ids", "cu_seqlens_q", "cu_seqlens_kv"})

    @property
    def non_tensor_names(self) -> frozenset[str]:
        return frozenset({"qkv_format"})

    @property
    def cp_sharded_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "position_ids"})

    @property
    def discriminating_names(self) -> frozenset[str]:
        return frozenset({"cu_seqlens_q", "cu_seqlens_kv", "qkv_format"})

    def has_required_names(self, names: set[str]) -> bool:
        return "input_ids" in names and "cu_seqlens_q" in names

    def detect_layout(self, raw: dict[int, dict[str, object]]) -> str:
        for step_data in raw.values():
            if (qkv_format := step_data.get("qkv_format")) is not None:
                fmt = qkv_format if isinstance(qkv_format, str) else str(qkv_format)
                if "bshd" in fmt.lower():
                    raise NotImplementedError(_BSHD_NOT_SUPPORTED_MSG)
                return "thd"

            input_ids = step_data.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
                raise NotImplementedError(_BSHD_NOT_SUPPORTED_MSG)

        warning_sink.add(
            GeneralWarning(
                category="layout_detection_fallback",
                message=(
                    "Megatron layout detection: no qkv_format or 2D input_ids found, "
                    "falling back to thd"
                ),
            )
        )
        return "thd"

    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: str, step: int
    ) -> TokenAlignerStepAux:
        input_ids: torch.Tensor = step_data["input_ids"]

        if (cu_seqlens_q := step_data.get("cu_seqlens_q")) is not None:
            seq_lens: torch.Tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        else:
            seq_lens = torch.tensor([input_ids.shape[0]], dtype=torch.long)

        if (position_ids := step_data.get("position_ids")) is not None:
            positions: torch.Tensor = position_ids
        else:
            positions = _infer_positions(seq_lens=seq_lens)

        seq_lens_list: list[int] = seq_lens.tolist()
        num_seqs: int = len(seq_lens_list)
        seq_ids: list[SeqId] = [
            PositionalSeqId(step=step, seq_index=seq_index)
            for seq_index in range(num_seqs)
        ]

        return TokenAlignerStepAux(
            input_ids=input_ids.tolist(),
            positions=positions.tolist(),
            seq_lens=seq_lens_list,
            seq_ids=seq_ids,
        )


# ── plugin registry ───────────────────────────────────────────────

_plugins: list[_AuxFrameworkPlugin] = [_SGLangPlugin(), _MegatronPlugin()]

AUX_NAMES: frozenset[str] = frozenset().union(*(p.all_names for p in _plugins))


# ── helpers ────────────────────────────────────────────────────────


def _infer_positions(*, seq_lens: torch.Tensor) -> torch.Tensor:
    """Infer positions when position_ids is missing (THD only)."""
    return torch.cat([torch.arange(int(slen.item())) for slen in seq_lens])
