"""Speculative target-verify extension of the linear-attention backends.

A speculative algorithm can replace the two state updates of a target verify
forward with its own kernels (for example faster kernels for its own draft
layout): the causal conv1d and, for GDN, the gated delta recurrence or, for
Mamba2, the selective state update. Its
``create_linear_attn_verify_extension(state)`` gets the backend's
``LinearAttnStateLayout`` (``kind`` names the mixer) and returns a
``LinearAttnVerifyExtension`` or None. The backend calls the extension at the
points below; every method defaults to "not handled", so the backend runs its
own kernels. Without an extension (every builtin) nothing changes.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention


@dataclasses.dataclass(frozen=True)
class LinearAttnStateLayout:
    """The backend's linear-attention state, as an extension sees it."""

    # Conv state width (conv kernel size - 1) and dtype.
    conv_state_len: int
    conv_dtype: torch.dtype
    # Recurrent state dtype and value heads per rank (Mamba2: heads per rank).
    ssm_dtype: torch.dtype
    num_v_heads: int
    # Per-step state snapshots (intermediate_ssm), which the post-verify
    # commit reads. Only that verify path calls the extension; a backend that
    # picks another path for a forward raises when an extension is set.
    has_state_snapshots: bool
    device: torch.device
    # The mixer: "gdn" or "mamba2".
    kind: str = "gdn"
    # Mamba2 only: head dim, SSM state size and B/C groups per rank.
    head_dim: Optional[int] = None
    ssm_state_size: Optional[int] = None
    n_groups: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class LinearAttnVerifyLayout:
    """The draft layout of a target-verify forward with tree links."""

    # [bs, width] tree links, -1 where absent.
    next_token: torch.Tensor
    next_sibling: torch.Tensor
    parent_token: torch.Tensor
    # [bs] state slots and [bs + 1] query starts.
    cache_indices: torch.Tensor
    query_start_loc: torch.Tensor
    width: int


class LinearAttnVerifyExtension:
    """Base extension: handles nothing. Subclass and override what you need.

    One instance serves the backend for its lifetime; it may keep its own
    static buffers (sized in ``init_cuda_graph_state``) and per-forward state
    (built in ``begin_forward``).
    """

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """After the backend sized its CUDA-graph buffers (same arguments)."""

    def begin_forward(
        self, layout: Optional[LinearAttnVerifyLayout], *, in_graph: bool
    ) -> None:
        """After the backend built the metadata of a forward, any mode.
        ``layout`` is set for a target verify with tree links, else None.
        ``in_graph=False`` for an eager forward; ``in_graph=True`` when the
        call is recorded into a CUDA graph and runs again on every replay."""

    def reads_strided_inputs(self) -> bool:
        """Whether the current forward's ``conv_update`` and
        ``recurrent_update`` take token-strided inputs: views of the input
        projection and of the conv output instead of split copies (``query``
        / ``key`` / ``value`` keep their ``(1, tokens, heads, dim)`` shape,
        ``a`` / ``b`` their ``(tokens, heads)`` shape). Asked after
        ``begin_forward``; default False."""
        return False

    def conv_update(
        self,
        layer: RadixLinearAttention | MambaMixer2,
        mixed_qkv: torch.Tensor,
        *,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_conv_window: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Target-verify conv1d. ``mixed_qkv`` is ``(batch, dim, width)``
        (Mamba2: the x / B / C projection before the conv). Return the conv
        output in the same layout, or None to run the backend's
        ``causal_conv1d_update``."""
        return None

    def recurrent_update(
        self,
        layer: RadixLinearAttention,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Target-verify recurrence. Must write the per-token states into
        ``intermediate_states`` like the backend's kernel, so the post-verify
        commit works unchanged. Return the attention output, or None to run
        the backend's ``target_verify`` kernel."""
        return None

    def ssm_update(
        self,
        layer: MambaMixer2,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        out: torch.Tensor,
        intermediate_states: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
    ) -> bool:
        """Mamba2 target-verify selective state update, with the inputs of
        the backend's ``selective_state_update`` call: ``x`` and ``dt``
        ``(batch, width, heads, head_dim)``, ``B`` and ``C`` ``(batch, width,
        n_groups, state)``, ``A`` ``(heads, head_dim, state)`` in fp32, ``D``
        and ``dt_bias`` ``(heads, head_dim)``; ``dt`` is before
        ``softplus(dt + dt_bias)``. Write the outputs into ``out`` ``(batch,
        width, heads, head_dim)`` and the per-token states into
        ``intermediate_states`` at its own per-slot step pitch, like the
        backend's kernel, so the post-verify commit works unchanged; return
        True. Return False to run the backend's kernel."""
        return False
