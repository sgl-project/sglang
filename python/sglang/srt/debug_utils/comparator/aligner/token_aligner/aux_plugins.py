from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    SeqId,
    SGLangSeqId,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.dims import TokenLayout
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink

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
    def detect_layout(self, raw: dict[int, dict[str, object]]) -> TokenLayout: ...

    @abstractmethod
    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: TokenLayout, step: int
    ) -> TokenAlignerStepAux: ...

    @abstractmethod
    def has_required_names(self, names: set[str]) -> bool:
        """Whether the minimum set of aux names needed for alignment is present."""
        ...

    @property
    def all_names(self) -> frozenset[str]:
        return self.tensor_names | self.non_tensor_names

    def extract_global_seq_lens(
        self, step_data: dict[str, object]
    ) -> Optional[list[int]]:
        """Extract per-seq token counts from loaded step data.

        Returns None if this framework doesn't support THD / no relevant data available.
        """
        return None

    def infer_cp_sharded_dims(self, name: str, ndim: int) -> str:
        """Infer dims string for a CP-sharded aux tensor based on its ndim."""
        raise NotImplementedError(
            f"infer_cp_sharded_dims not implemented for {type(self).__name__}"
        )


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

    def detect_layout(self, raw: dict[int, dict[str, object]]) -> TokenLayout:
        return TokenLayout.T

    def extract_global_seq_lens(
        self, step_data: dict[str, object]
    ) -> Optional[list[int]]:
        if not self.cp_sharded_names:
            return None

        seq_lens = step_data.get("seq_lens")
        if not isinstance(seq_lens, torch.Tensor):
            return None

        return seq_lens.tolist()

    def infer_cp_sharded_dims(self, name: str, ndim: int) -> str:
        """Infer dims for CP-sharded aux tensors.

        NOTE: assumes zigzag ordering — natural-order CP without explicit dims
        will be mishandled. Callers should set dims explicitly for non-zigzag CP.
        """
        if ndim == 1:
            return "t(cp,zigzag)"
        raise ValueError(
            f"SGLang: cannot infer dims for CP-sharded '{name}' with ndim={ndim}"
        )

    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: TokenLayout, step: int
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
        return "input_ids" in names

    def extract_global_seq_lens(
        self, step_data: dict[str, object]
    ) -> Optional[list[int]]:
        if not self.cp_sharded_names:
            return None

        cu_seqlens_q = step_data.get("cu_seqlens_q")
        if not isinstance(cu_seqlens_q, torch.Tensor):
            return None

        return (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()

    def infer_cp_sharded_dims(self, name: str, ndim: int) -> str:
        """Infer dims for CP-sharded aux tensors.

        NOTE: assumes zigzag ordering — natural-order CP without explicit dims
        will be mishandled. Callers should set dims explicitly for non-zigzag CP.
        """
        if ndim == 1:
            return "t(cp,zigzag)"
        if ndim == 2:
            return "b s(cp,zigzag)"
        raise ValueError(
            f"Megatron: cannot infer dims for CP-sharded '{name}' with ndim={ndim}"
        )

    def detect_layout(self, raw: dict[int, dict[str, object]]) -> TokenLayout:
        for step_data in raw.values():
            if (qkv_format := step_data.get("qkv_format")) is not None:
                fmt = qkv_format if isinstance(qkv_format, str) else str(qkv_format)
                if "bshd" in fmt.lower():
                    return TokenLayout.BS
                return TokenLayout.T

            input_ids = step_data.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
                return TokenLayout.BS

        warning_sink.add(
            GeneralWarning(
                category="layout_detection_fallback",
                message=(
                    "Megatron layout detection: no qkv_format or 2D input_ids found, "
                    "falling back to T"
                ),
            )
        )
        return TokenLayout.T

    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: TokenLayout, step: int
    ) -> TokenAlignerStepAux:
        input_ids: torch.Tensor = step_data["input_ids"]
        is_bshd: bool = layout == TokenLayout.BS

        # BSHD [B, S] → flat [B*S]; THD [T] stays as-is
        flat_ids: list[int] = input_ids.reshape(-1).tolist()

        if (cu_seqlens_q := step_data.get("cu_seqlens_q")) is not None:
            seq_lens_list: list[int] = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        elif is_bshd:
            seq_lens_list = [input_ids.shape[1]] * input_ids.shape[0]
        else:
            seq_lens_list = [input_ids.shape[0]]

        if (position_ids := step_data.get("position_ids")) is not None:
            flat_positions: list[int] = position_ids.reshape(-1).tolist()
        elif is_bshd:
            flat_positions = list(range(input_ids.shape[1])) * input_ids.shape[0]
        else:
            flat_positions = _infer_positions(
                seq_lens=torch.tensor(seq_lens_list)
            ).tolist()

        num_seqs: int = len(seq_lens_list)
        seq_ids: list[SeqId] = [
            PositionalSeqId(step=step, seq_index=seq_index)
            for seq_index in range(num_seqs)
        ]

        return TokenAlignerStepAux(
            input_ids=flat_ids,
            positions=flat_positions,
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
