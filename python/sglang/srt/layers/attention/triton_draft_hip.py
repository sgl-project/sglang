"""ROCm draft attention with BF16 dots and rank-local DCP sliding windows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.mla_window_indices import mla_window_sparse_indices
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


def init_draft_backend(backend: TritonAttnBackend, model_runner: ModelRunner) -> None:
    """Configure a ROCm draft worker's Triton backend once its DCP size is final."""
    algorithm = model_runner.spec_algorithm
    # DFLASH / DSPARK draft KV follows --kv-cache-dtype unless
    # --speculative-draft-kv-cache-dtype says otherwise; keep q and the attention
    # math in BF16 over it instead of casting q and the probabilities to FP8.
    backend.draft_bf16_dot = algorithm.is_dflash_family()
    # A DFLASH draft never runs forward_decode, so its graphs skip the split-KV
    # decode partials (all heads under DCP: ~14.5 GiB a rank for K3.1's MLA5 draft).
    backend.runs_decode = not algorithm.is_dflash()
    # The monokernel's DFLASH MLA draft verifies on its own tuned copy of the
    # grouped-head kernel (a sliding window, two heads a tile); nothing else does.
    if (
        algorithm.is_dflash()
        and backend.use_mla
        and backend.topk == 1
        and is_gfx95_supported()
        and get_exec().kernel.enable_amd_monokernel
        and model_runner.model_config.is_dflash_mla_draft
    ):
        from sglang.kernels.ops.attention.verify_mla_draft import verify_shared_kv_fwd

        backend.verify_mla_draft_fwd = torch.compiler.disable(verify_shared_kv_fwd)
    if algorithm.is_dflash() and backend.use_mla and backend.dcp_size > 1:
        backend.mla_dcp_window_verify = MlaDcpWindowVerify(backend, model_runner)
        # Each layer applies its own window there, so the backend builds no
        # window rail.
        backend.sliding_window_size = None


def check_dcp_mla_draft(draft_hf_config, draft_window_size: Optional[int]) -> None:
    """Refuse the DFLASH MLA drafts this backend cannot serve under DCP."""
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_layer_types,
        is_dflash_mla_draft_config,
    )

    if not is_dflash_mla_draft_config(draft_hf_config):
        return
    # Only the ROCm MLA forward (SGLANG_USE_AITER) hands this backend a DCP verify
    # window with gathered heads and merges its rank-local partials; it reads owner
    # rows through the shared req_to_token, not a compact draft table.
    if not envs.SGLANG_USE_AITER.get() or draft_window_size is not None:
        raise ValueError(
            "On ROCm, DFLASH mla drafts run on the triton draft attention backend "
            "under --dcp-size > 1 only with SGLANG_USE_AITER=1 and without "
            "--speculative-draft-window-size; otherwise serve them without DCP."
        )
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        raise ValueError(
            "The triton draft attention backend serves only sliding_attention "
            f"DFLASH mla drafts under --dcp-size > 1 (layer_types={layer_types}); "
            "serve full-attention drafts without DCP."
        )


class MlaDcpWindowVerify:
    """Attend to this rank's keys in [p - window, p] and return partials for DCP.

    Read window rows from req_to_token or the draft ring, without a global prefix rail.
    """

    def __init__(self, backend: TritonAttnBackend, model_runner: ModelRunner):
        self.backend = backend
        # The DFLASH worker attaches the draft KV ring after the backends exist.
        self.draft_runner = model_runner
        # Built at the first sliding layer of a forward and reused by the next ones:
        # (forward_batch, layer_id, window, indices, counts).
        self._index_cache = None

    def window_sparse_indices(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse rank-local window indices across consecutive matching draft layers."""
        cache = self._index_cache
        if (
            cache is not None
            and cache[0] is forward_batch
            and cache[1] == layer.layer_id - 1
            and cache[2] == layer.sliding_window_size
        ):
            indices, counts = cache[3], cache[4]
        else:
            backend = self.backend
            ring = self.draft_runner.draft_kv_ring
            indices, counts = mla_window_sparse_indices(
                req_to_token=backend.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                positions=forward_batch.positions,
                window_left=layer.sliding_window_size,
                dcp_size=backend.dcp_size,
                dcp_rank=backend.dcp_rank,
                page_size=backend.page_size,
                v2p=None if ring else backend.kv_index_translator.full_v2p_table,
                ring_rows_per_req=ring.rows_per_req if ring else 0,
                min_pos=ring.min_pos if ring else None,
            )
        self._index_cache = (
            forward_batch,
            layer.layer_id,
            layer.sliding_window_size,
            indices,
            counts,
        )
        return indices, counts

    def forward(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-local (out, base-2 lse) of one layer; the window's own keys are
        read back from the cache, where the caller wrote them."""
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
            triton_sparse_mla_fwd,
        )

        if not (causal and forward_batch.forward_mode.is_target_verify()):
            raise NotImplementedError(
                "DCP Triton extend applies a sliding window only to causal MLA "
                "target-verify forwards"
            )
        indices, _ = self.window_sparse_indices(layer, forward_batch)
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        out, lse = triton_sparse_mla_fwd(
            q_nope=q[..., : layer.v_head_dim],
            q_rope=q[..., layer.v_head_dim :],
            kv=self.backend.token_to_kv_pool.get_key_buffer(layer.layer_id),
            indices=indices.unsqueeze(1),
            sm_scale=layer.scaling,
            d_v=layer.v_head_dim,
            return_lse=True,
            # A raw FP8 cache is upcast; q stays BF16.
            bf16_dot=True,
        )
        # The kernel's LSE is natural log and the merge of Triton partials is base 2
        # (is_mla_dcp_lse_base_on_e); -inf rows (no owned key) stay -inf.
        return out.squeeze(0).flatten(1), lse * math.log2(math.e)
