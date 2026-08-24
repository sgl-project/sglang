from __future__ import annotations

import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.token_probe.paged_attention import (
    MAX_SPLITS,
    probe_paged_attention,
)
from sglang.srt.models.token_probe.probe_kernels import tap_into
from sglang.srt.models.token_probe.loader import load_probe_head
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args


def _same_model_path(lhs: Optional[str], rhs: Optional[str]) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if os.path.exists(lhs) or os.path.exists(rhs):
        return os.path.realpath(lhs) == os.path.realpath(rhs)
    return lhs.rstrip("/") == rhs.rstrip("/")


def _is_mtp_spec(server_args) -> bool:
    """Whether the resolved speculative configuration is bundled MTP.

    SGLang resolves the public ``NEXTN`` spelling to ``EAGLE`` before model
    construction.  A linear chain (top-k 1) backed by the target checkpoint is
    the remaining distinction from an external EAGLE draft model.
    """
    algorithm = (server_args.speculative_algorithm or "").upper()
    if algorithm != "EAGLE" or server_args.speculative_eagle_topk != 1:
        return False
    draft_path = server_args.speculative_draft_model_path
    return draft_path is None or _same_model_path(draft_path, server_args.model_path)


class TokenProbe(nn.Module):
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = False
        self.last_scores = None
        self.probe_model: Optional[nn.Module] = None
        # Tapped layer id -> its slot in the feature row; read per decoder
        # layer, so resolved once at load.
        self.tap_slots: dict = {}
        # Per-forward: the [tokens, num_taps * hidden] buffer the taps write
        # into, how many landed, and whether finalize already launched.
        self._features_buf = None
        self._captured = 0
        self._launched = False
        self._scores_leave_this_rank = False
        self.prefill_enabled = envs.SGLANG_ENABLE_TOKEN_PROBE_PREFILL.get()
        self._base_hidden_size = config.hidden_size
        self._hc_mult = getattr(config, "hc_mult", 1)

        # For multi-stream overlap
        self.overlap = envs.SGLANG_ENABLE_TOKEN_PROBE_OVERLAP.get()
        self._side_stream = None
        self._pending_event = None
        self._pending_scores = None

        # Attention-head K/V, addressed by the base model's KV slot ids so the
        # pool inherits the base allocator's slot lifecycle.
        self._kv_pool = None
        self._req_to_token = None
        self._decode_qo_indptr = None
        self._warned_missing_kv_pool = False
        self._warned_untapped = False
        self._pending_verify = None
        self._verify_score_event = None

        # Per-forward scratch captured in begin_forward, consumed in finalize.
        self._is_extend = False
        self._is_verify = False
        self._out_cache_loc = None
        self._req_indices = None
        self._positions = None
        self._seq_lens_cpu = None
        self._extend_seq_lens = None
        self._extend_seq_lens_cpu = None

        ckpt_path = ckpt_path or self._probe_ckpt_path()
        if not ckpt_path:
            return

        self.probe_model = load_probe_head(ckpt_path, dtype=torch.get_default_dtype())
        self._validate_checkpoint(config, ckpt_path)
        if get_global_server_args().enable_attn_tp_input_scattered:
            # Taps read the layer loop's tensors, which that flag turns into
            # this rank's token shard -- the probe would silently score a
            # fraction of the batch. (Rejected on the flag rather than on the
            # conditions it also needs, so the failure cannot be silent.)
            raise ValueError(
                "The token probe cannot run with "
                "--enable-attn-tp-input-scattered: the decoder layers would "
                "hand it only this rank's slice of the tokens."
            )
        server_args = get_global_server_args()
        spec_algorithm = server_args.speculative_algorithm
        if spec_algorithm is not None and not _is_mtp_spec(server_args):
            raise ValueError(
                "Token probe speculative decoding currently supports only bundled "
                "MTP/NEXTN (internally resolved to EAGLE with top-k 1 and the "
                "target checkpoint as the draft checkpoint); got "
                f"algorithm={spec_algorithm!r}, "
                f"topk={server_args.speculative_eagle_topk!r}, "
                f"draft_model={server_args.speculative_draft_model_path!r}. "
                "EAGLE, EAGLE3, DFlash, and Standalone are not supported."
            )

        self.tap_slots = {
            layer_id: slot
            for slot, layer_id in enumerate(self.probe_model.state_indices)
        }
        self._scores_leave_this_rank = get_parallel().attn_tp_rank == 0
        self.enabled = True
        self.logger.info(
            "Loaded SGLang token probe (%s) from %s",
            type(self.probe_model).__name__,
            ckpt_path,
        )

    @property
    def uses_kv_cache(self) -> bool:
        return self.probe_model is not None and hasattr(self.probe_model, "kv_dim")

    @staticmethod
    def _probe_ckpt_path() -> Optional[str]:
        return get_global_server_args().probe_ckpt

    def _validate_checkpoint(self, config, ckpt_path: str) -> None:
        assert self.probe_model is not None
        # A head that declares no hidden size (the identity probe) adopts the
        # base model's.
        probe_hidden = self.probe_model.hidden_size
        assert probe_hidden is None or probe_hidden == config.hidden_size, (
            f"token probe hidden size {probe_hidden} from "
            f"{ckpt_path} does not match model hidden size {config.hidden_size}"
        )
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            return
        invalid = [
            layer_id
            for layer_id in self.probe_model.state_indices
            if layer_id < 0 or layer_id >= num_hidden_layers
        ]
        if invalid:
            raise ValueError(
                "token probe base_model_layer_ids are outside the base model's "
                f"[0, {num_hidden_layers}) layer range: {invalid}"
            )

    def prepare_device(self, device: torch.device) -> None:
        if self.probe_model is None:
            return None
        if device.type != "cuda":
            return None

        self.probe_model.to(device)
        self.probe_model.eval()
        self._warmup_side_kernels(device)

    def _warmup_side_kernels(self, device: torch.device) -> None:
        # The tap and classifier kernels JIT on their first call like the
        # attention one, and the mlp head never reaches _warmup_attn_kernel
        # (no K/V pool), so warm them from here, where every head lands.
        dtype = torch.get_default_dtype()
        hidden = self.probe_model.hidden_size or self._base_hidden_size
        if hidden and self.tap_slots:
            features = torch.empty(
                1, len(self.tap_slots) * hidden, device=device, dtype=dtype
            )
            states = torch.zeros(1, hidden, device=device, dtype=dtype)
            # HAS_RESIDUAL is constexpr, and models differ: gemma4 folds the
            # residual into the layer and taps with None.
            for residual in (states, None):
                tap_into(
                    features=features,
                    slot=0,
                    hidden_states=states,
                    residual=residual,
                )
            if self._hc_mult > 1:
                tap_into(
                    features=features,
                    slot=0,
                    hidden_states=torch.zeros(
                        1,
                        self._hc_mult,
                        hidden,
                        device=device,
                        dtype=dtype,
                    ),
                    residual=None,
                )
        if self.uses_kv_cache:
            width = self.probe_model.q_dim
            dtype = self.probe_model.kv_dtype
            rows = torch.zeros(1, width, device=device, dtype=dtype)
            self.probe_model.classify(rows, rows)

    def init_overlap(self, device: torch.device) -> None:
        if self.probe_model is None or not self.overlap:
            return None
        if device.type != "cuda":
            return None

        self._side_stream = torch.cuda.Stream(device=device)

    def init_kv_cache(self, num_slots: int, req_to_token: torch.Tensor) -> None:
        if not self.uses_kv_cache or self._kv_pool is not None:
            return
        device = next(self.probe_model.parameters()).device
        kv_dim = self.probe_model.kv_dim
        dtype = self.probe_model.kv_dtype
        # K and V share one pool row: the fused qkv projection already emits
        # them adjacent, so publishing a token costs a single index_copy_ and
        # the attention kernel reads both from the same row.
        self._kv_pool = torch.zeros(num_slots, 2 * kv_dim, device=device, dtype=dtype)
        self._req_to_token = req_to_token
        self.logger.info(
            "token probe KV pool: %d slots x (K+V) x 1 head x %d dims, %.2f GB",
            num_slots,
            self.probe_model.head_dim,
            self._kv_pool.numel() * self._kv_pool.element_size() / 1e9,
        )
        self._warmup_attn_kernel(device)

    def _warmup_attn_kernel(self, device: torch.device) -> None:
        # Compile every variant now: triton JITs on first call, and on a live
        # server that lands inside a forward, stalling the scheduler loop
        # (~90ms per variant, per rank) while queued requests pile up.
        # BLOCK_M and NUM_SPLITS are both constexpr, so each (block size, split
        # count) pair is its own compilation -- warming one split count left
        # prefill's single-pass launch to JIT under load, which is what showed
        # up as a 17-30% TTFT regression that faded as concurrency rose.
        splits = [1]
        while splits[-1] < MAX_SPLITS:
            splits.append(splits[-1] * 2)
        for max_qo_len in (1, 64):
            rows = max(max_qo_len, 1)
            for force_splits in splits:
                probe_paged_attention(
                    q=torch.zeros(
                        rows,
                        self.probe_model.q_dim,
                        device=device,
                        dtype=self._kv_pool.dtype,
                    ),
                    kv_pool=self._kv_pool,
                    req_to_token=self._req_to_token,
                    req_indices=torch.zeros(1, dtype=torch.int64, device=device),
                    positions=torch.zeros(rows, dtype=torch.int64, device=device),
                    qo_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
                    max_qo_len=max_qo_len,
                    num_heads=self.probe_model.num_attention_heads,
                    head_dim=self.probe_model.head_dim,
                    window=self.probe_model.sliding_window,
                    force_splits=force_splits,
                )

    def reset_last_scores(self) -> None:
        self.last_scores = None

    def begin_forward(
        self, forward_batch: ForwardBatch, hidden_states: torch.Tensor
    ) -> bool:
        # True when the decoder layers should tap this forward. Allocates the
        # row buffer the taps write into.
        self.reset_last_scores()
        self._is_extend = False
        self._is_verify = False
        self._launched = False
        self._captured = 0
        self._features_buf = None
        if not self._begin_forward(forward_batch, hidden_states):
            return False
        self._features_buf = torch.empty(
            hidden_states.shape[0],
            len(self.tap_slots) * hidden_states.shape[-1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return True

    def _begin_forward(
        self, forward_batch: ForwardBatch, hidden_states: torch.Tensor
    ) -> bool:
        if not self.enabled or not self._scores_leave_this_rank:
            return False
        if self._verify_score_event is not None:
            # In-flight side-stream verify scoring may still write the pool.
            torch.cuda.current_stream().wait_event(self._verify_score_event)

        if forward_batch.forward_mode.is_decode():
            if min(int(forward_batch.batch_size), hidden_states.shape[0]) == 0:
                return False
            if self.uses_kv_cache and self._kv_pool is None:
                self._warn_missing_kv_pool()
                return False
            self._out_cache_loc = forward_batch.out_cache_loc
            self._req_indices = forward_batch.req_pool_indices
            self._positions = forward_batch.positions
            self._seq_lens_cpu = forward_batch.seq_lens_cpu
            return True

        if forward_batch.forward_mode.is_extend():
            # MIXED interleaves decode rows into the extend layout and would
            # break the per-request offset mapping.
            if forward_batch.forward_mode.is_mixed():
                return False
            if forward_batch.forward_mode.is_target_verify():
                # The attention head only projects q/k/v here; the verify
                # worker scores the accepted chain via score_accepted_verify.
                if self.uses_kv_cache:
                    if self._kv_pool is None:
                        self._warn_missing_kv_pool()
                        return False
                    self._is_verify = True
                return True
            if hidden_states.shape[0] == 0:
                return False
            if self.uses_kv_cache:
                if self._kv_pool is None:
                    self._warn_missing_kv_pool()
                    return False
                # The chunk's K/V must be written even with prefill scoring
                # off; _dispatch_compute gates whether scores are surfaced.
                self._is_extend = True
                self._out_cache_loc = forward_batch.out_cache_loc
                self._req_indices = forward_batch.req_pool_indices
                self._positions = forward_batch.positions
                self._seq_lens_cpu = forward_batch.seq_lens_cpu
                self._extend_seq_lens = forward_batch.extend_seq_lens
                self._extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
                return True
            return self.prefill_enabled

        return False

    def taps_within(self, start: int, end: int) -> bool:
        """True when every tapped layer runs in ``[start, end)``.

        Models whose loop hands part of the range to another path (two-batch
        overlap, pipeline splits) use this to skip the probe rather than tap
        some layers and miss the rest.
        """
        return all(start <= layer_id < end for layer_id in self.tap_slots)

    def capture(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> None:
        """Tap a decoder layer's input residual stream, if it is tapped.

        Call this in the layer loop before the layer runs, which is both where
        the value is available and where the overlap window is longest. The
        tap is normalized and stored straight into the feature buffer, so it
        survives the layer overwriting its residual in place.

        The loop carries ScatterMode.TP_ATTN_FULL, so each rank holds its
        attention group's whole token set -- under dp attention that is the
        group's sequences, which is exactly what its rank 0 emits. Only
        --enable-attn-tp-input-scattered would narrow this to a token shard,
        and __init__ rejects that combination.
        """
        slot = self.tap_slots.get(layer_id)
        if slot is None or self._features_buf is None:
            return
        tap_into(self._features_buf, slot, hidden_states, residual)
        self._captured += 1
        if (
            self.overlap
            and not self._launched
            and self._captured == len(self.tap_slots)
        ):
            self._launched = True
            self.finalize_async()

    def finish(self) -> Optional[torch.Tensor]:
        """Collect the scores, joining the overlapped finalize if one ran."""
        if self._launched:
            return self.finalize_join()
        return self.finalize()

    def _warn_missing_kv_pool(self) -> None:
        if self._warned_missing_kv_pool:
            return
        self._warned_missing_kv_pool = True
        self.logger.error(
            "attention token probe has no KV pool (init_kv_cache was not "
            "called for this worker); the probe is skipped."
        )

    def _ready(self) -> bool:
        if self._features_buf is None:
            return False
        if self._captured == 0:
            # A forward that bypassed the tapped loop entirely, e.g. two-batch
            # overlap. Skip it rather than take the server down.
            self._warn_untapped()
            return False
        expected = len(self.tap_slots)
        assert self._captured == expected, (
            f"token probe captured {self._captured} hidden states, expected "
            f"{expected}: the model's decoder layers are not tapping every "
            f"layer in base_model_layer_ids."
        )
        return True

    def _warn_untapped(self) -> None:
        if self._warned_untapped:
            return
        self._warned_untapped = True
        self.logger.warning(
            "token probe saw a forward with no taps (a path that skips the "
            "decoder layer loop, such as two-batch overlap); it is scoring "
            "nothing for those forwards."
        )

    @property
    def label_names(self) -> tuple:
        if self.probe_model is None:
            return ()
        return tuple(self.probe_model.label_names)

    @staticmethod
    def _as_scores(probe_score: torch.Tensor) -> torch.Tensor:
        return probe_score.float().detach()

    def _compute_scores(self) -> torch.Tensor:
        probe_score = self.probe_model.forward_features(
            self._features_buf, aggregate=False
        )["probe_score"]
        return self._as_scores(probe_score)

    def _compute_scores_kv(self, write_only: bool):
        qkv = self.probe_model.project_fused(self._features_buf)
        q_dim = self.probe_model.q_dim
        loc = self._out_cache_loc.long()
        self._kv_pool.index_copy_(0, loc, qkv[:, q_dim:].to(self._kv_pool.dtype))
        if write_only:
            return None

        q = qkv[:, :q_dim]
        rows = self._attn_rows(q)
        probe_score = self.probe_model.classify(rows, q)
        return self._as_scores(probe_score)

    def _attn_rows(self, q: torch.Tensor) -> torch.Tensor:
        if self._is_extend:
            # Both of these must stay off the host: finalize_async runs this
            # on the side stream after wait_stream(main), so a pageable H2D
            # copy here blocks the CPU until the whole forward has drained,
            # turning the overlapped probe into a synchronization barrier.
            # (Measured: 634ms vs 481ms median TTFT at concurrency 2.)
            qo_indptr = F.pad(torch.cumsum(self._extend_seq_lens, 0), (1, 0)).to(
                torch.int32
            )
            max_qo_len = int(max(self._extend_seq_lens_cpu))
        else:
            # Decode's indptr is just 0..bs; keep one grown buffer instead of
            # launching an arange every step.
            n = q.shape[0] + 1
            if self._decode_qo_indptr is None or self._decode_qo_indptr.numel() < n:
                self._decode_qo_indptr = torch.arange(
                    max(n, 512), dtype=torch.int32, device=q.device
                )
            qo_indptr = self._decode_qo_indptr[:n]
            max_qo_len = 1
        return probe_paged_attention(
            q=q,
            kv_pool=self._kv_pool,
            req_to_token=self._req_to_token,
            req_indices=self._req_indices,
            positions=self._positions,
            qo_indptr=qo_indptr,
            max_qo_len=max_qo_len,
            num_heads=self.probe_model.num_attention_heads,
            head_dim=self.probe_model.head_dim,
            window=self.probe_model.sliding_window,
        )

    def _dispatch_compute(self) -> Optional[torch.Tensor]:
        if self.uses_kv_cache:
            if self._is_verify:
                qkv = self.probe_model.project_fused(self._features_buf)
                return qkv.detach()

            write_only = self._is_extend and not self.prefill_enabled
            return self._compute_scores_kv(write_only=write_only)
        return self._compute_scores()

    def score_accepted_verify(
        self,
        *,
        qkv: torch.Tensor,
        forward_batch,
        num_accept_slots: int,
        num_draft_tokens: int,
    ) -> Optional[torch.Tensor]:
        if self._kv_pool is None or qkv.shape[0] == 0:
            return None
        pending = self._verify_attn_partial(
            qkv=qkv,
            forward_batch=forward_batch,
            num_accept_slots=num_accept_slots,
            num_draft_tokens=num_draft_tokens,
        )
        return self._finish_verify_scores(pending)

    def begin_score_accepted_verify(
        self,
        *,
        qkv: torch.Tensor,
        forward_batch,
        num_accept_slots: int,
        num_draft_tokens: int,
    ) -> bool:
        if self._side_stream is None or self._kv_pool is None or qkv.shape[0] == 0:
            return False
        main = torch.cuda.current_stream()
        self._side_stream.wait_stream(main)  # see qkv, acceptance, prior pool state
        with torch.cuda.stream(self._side_stream):
            pending = self._verify_attn_partial(
                qkv=qkv,
                forward_batch=forward_batch,
                num_accept_slots=num_accept_slots,
                num_draft_tokens=num_draft_tokens,
            )
            self._pending_verify = self._finish_verify_scores(pending)
        self._verify_score_event = torch.cuda.Event()
        self._verify_score_event.record(self._side_stream)
        return True

    def finish_score_accepted_verify(self) -> Optional[torch.Tensor]:
        if self._verify_score_event is not None:
            torch.cuda.current_stream().wait_event(self._verify_score_event)
            self._verify_score_event = None
        pending, self._pending_verify = self._pending_verify, None
        if pending is None:
            return None
        if isinstance(pending, tuple):
            return self._finish_verify_scores(pending)
        return pending

    def _verify_attn_partial(
        self,
        *,
        qkv: torch.Tensor,
        forward_batch,
        num_accept_slots: int,
        num_draft_tokens: int,
    ) -> tuple:
        # Publish the block-front rows' K/V at their chain slots and attend
        # over the committed prefix plus the earlier chain rows. The returned
        # attention rows and Q residuals are classified together afterward.
        nd = num_draft_tokens
        bs = qkv.shape[0] // nd
        s1 = min(num_accept_slots, nd)
        heads = self.probe_model.num_attention_heads
        head_dim = self.probe_model.head_dim
        d = self.probe_model.q_dim
        row_dim = d + 2 * head_dim
        rows = qkv.view(bs, nd, row_dim)[:, :s1]
        q, k, v = rows.split([d, head_dim, head_dim], dim=-1)
        loc = forward_batch.out_cache_loc.view(bs, nd)[:, :s1].reshape(-1).long()
        self._kv_pool.index_copy_(
            0,
            loc,
            torch.cat((k, v), dim=-1).reshape(-1, 2 * head_dim).to(self._kv_pool.dtype),
        )

        # Verify-batch seq_lens are the committed pre-verify lengths.
        prefix_lens = forward_batch.seq_lens.view(bs, 1, 1)
        window = min(
            self._req_to_token.shape[1],
            max(int(forward_batch.seq_lens_cpu.max()), 1),
        )
        table = self._req_to_token.index_select(
            0, forward_batch.req_pool_indices.long()
        )[:, :window].long()
        prefix = self._kv_pool[table]
        pk = prefix[..., :head_dim].view(bs, window, 1, head_dim).transpose(1, 2)
        pv = prefix[..., head_dim:].view(bs, window, 1, head_dim).transpose(1, 2)
        ck = k.reshape(bs, s1, 1, head_dim).transpose(1, 2)
        cv = v.reshape(bs, s1, 1, head_dim).transpose(1, 2)
        keys = torch.cat([pk, ck], dim=2)
        values = torch.cat([pv, cv], dim=2)

        device = qkv.device
        steps = torch.arange(s1, device=device)
        key_pos = torch.arange(window, device=device).view(1, 1, -1)
        prefix_mask = (key_pos < prefix_lens).expand(bs, s1, window)
        chain_mask = (
            (steps.view(1, -1) <= steps.view(-1, 1)).view(1, s1, s1).expand(bs, s1, s1)
        )
        sliding_window = self.probe_model.sliding_window
        if sliding_window is not None:
            query_pos = prefix_lens + steps.view(1, s1, 1)
            prefix_mask = prefix_mask & (key_pos > query_pos - sliding_window)
            chain_mask = chain_mask & (
                steps.view(1, -1) > steps.view(-1, 1) - sliding_window
            ).view(1, s1, s1)
        mask = torch.cat([prefix_mask, chain_mask], dim=-1).unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q.reshape(bs, s1, heads, head_dim).transpose(1, 2),
            keys.expand(bs, heads, keys.shape[2], head_dim),
            values.expand(bs, heads, values.shape[2], head_dim),
            attn_mask=mask,
            scale=1.0 / math.sqrt(head_dim),
        )
        rows_out = out.transpose(1, 2).reshape(bs * s1, d)
        return (rows_out, q.reshape(bs * s1, d), bs, nd, s1)

    def _finish_verify_scores(self, pending: tuple) -> torch.Tensor:
        # Reduce the rank-local logits and scatter the chain rows into the
        # [bs * nd] block-front layout the scheduler reads.
        rows_out, q, bs, nd, s1 = pending
        probs = self.probe_model.classify(rows_out, q)
        scores = probs.new_zeros(bs * nd, probs.shape[-1])
        block = (torch.arange(bs, device=probs.device) * nd).view(bs, 1) + torch.arange(
            s1, device=probs.device
        )
        scores[block.reshape(-1)] = probs
        return scores.detach()

    def finalize(self) -> Optional[torch.Tensor]:
        if not self._ready():
            return None
        self.last_scores = self._dispatch_compute()
        return self.last_scores

    def finalize_async(self) -> None:
        if self._side_stream is None or not self._ready():
            self._pending_event = None
            self._launched = False
            self.finalize()
            return
        main = torch.cuda.current_stream()
        self._side_stream.wait_stream(main)
        with torch.cuda.stream(self._side_stream):
            self._pending_scores = self._dispatch_compute()
        self._pending_event = torch.cuda.Event()
        self._pending_event.record(self._side_stream)

    def finalize_join(self) -> Optional[torch.Tensor]:
        if self._pending_event is not None:
            torch.cuda.current_stream().wait_event(self._pending_event)
            self.last_scores = self._pending_scores
            self._pending_event = None
            self._pending_scores = None
        return self.last_scores
