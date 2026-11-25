"""Fixed-capacity FlashInfer KDA plans for breakable prefill graphs."""

import numpy as np
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.kda_input_pack import contiguous_kda_inputs
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
    flashinfer_kda_min_engine_seqs,
)


class FlashInferKDAPrefillGraph:
    def __init__(self, *, wrapper_cls, num_tokens, max_seqs, num_heads, track, device):
        self.wrapper_cls = wrapper_cls
        # A workspace can be captured only once: each layer owns its plan.
        self.wrappers = {}
        self.route_seqs = max(
            max_seqs,
            min(num_tokens, flashinfer_kda_min_engine_seqs(num_heads, device)),
        )
        self.cu_seqlens = torch.empty(
            self.route_seqs + 1, dtype=torch.int64, device=device
        )
        self.cache_indices = torch.empty(
            self.route_seqs, dtype=torch.int32, device=device
        )
        self.track = track
        if track:
            self.checkpoint_cu_starts = torch.empty_like(self.cu_seqlens)
            self.checkpoint_indices = torch.full(
                (num_tokens // 64,), -1, dtype=torch.int32, device=device
            )

    def fill(self, meta, extend_seq_lens, track_rows):
        self.cu_seqlens[: meta.max_seqs + 1].copy_(meta.cu_seqlens)
        self.cu_seqlens[meta.max_seqs + 1 :].fill_(sum(extend_seq_lens))
        # Empty rows may read and write slot 0, SGLang's reserved dummy write
        # sink. FlashInfer does not accept Triton's negative state-slot sentinel.
        self.cache_indices[: meta.max_seqs].copy_(meta.cache_indices.clamp_min(0))
        self.cache_indices[meta.max_seqs :].zero_()
        if self.track:
            counts = np.asarray(extend_seq_lens, dtype=np.int64) // 64
            starts = np.empty(self.cu_seqlens.numel(), dtype=np.int64)
            starts[0] = 0
            np.cumsum(counts, out=starts[1 : len(counts) + 1])
            starts[len(counts) + 1 :] = starts[len(counts)]
            indices = np.full(self.checkpoint_indices.numel(), -1, dtype=np.int32)
            if track_rows is not None:
                for row, chunk in track_rows.chunk_rows:
                    # Triton names the state before chunk; FlashInfer snapshots
                    # after each interval. Write that snapshot directly to row.
                    indices[starts[row] + chunk - 1] = row
            self.checkpoint_cu_starts.copy_(
                torch.from_numpy(starts).pin_memory(), non_blocking=True
            )
            self.checkpoint_indices.copy_(
                torch.from_numpy(indices).pin_memory(), non_blocking=True
            )
        for wrapper in self.wrappers.values():
            wrapper.plan(self.cu_seqlens)

    def run(self, *, layer, q, k, v, g, beta, ssm_states, h_track):
        num_tokens = q.shape[1]
        if num_tokens <= self.route_seqs:
            # Plain prefill requires T > route_seqs; the trailing token is
            # outside every cu_seqlens row.
            q, k, v, g, beta = (
                F.pad(x, (0, 0) * (x.ndim - 2) + (0, 1)) for x in (q, k, v, g, beta)
            )
        wrapper = self.wrappers.get(layer.layer_id)
        if wrapper is None:
            wrapper = self.wrapper_cls(q.device)
            wrapper.plan(self.cu_seqlens)
            self.wrappers[layer.layer_id] = wrapper
        checkpoints = {}
        if self.track:
            checkpoints = dict(
                state_checkpoints=h_track,
                checkpoint_cu_starts=self.checkpoint_cu_starts,
                checkpoint_state_indices=self.checkpoint_indices,
                checkpoint_every_n_tokens=64,
            )
        q, k, v, g, beta = contiguous_kda_inputs(q, k, v, g, beta)
        return wrapper.run(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=layer.A_log.reshape(-1).float().contiguous(),
            dt_bias=layer.dt_bias.reshape(-1).float().contiguous(),
            initial_state=ssm_states,
            ssm_state_indices=self.cache_indices,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=layer.lower_bound,
            beta_is_logit=True,
            output=torch.empty_like(q),
            **checkpoints,
        )[0][:, :num_tokens]
