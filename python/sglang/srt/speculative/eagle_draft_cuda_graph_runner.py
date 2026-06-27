from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    ShapeKey,
    _grouped_foreach_copy_,
    get_batch_sizes_to_capture,
    model_capture_mode,
)
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    maybe_flashinfer_autotune_speculative_draft,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.runtime_context import get_flags
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import get_draft_recurrent_hidden_state_spec
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.async_probe import maybe_detect_nan, maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


@dataclass
class EagleDraftInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    rids_int: Optional[torch.Tensor]
    bootstrap_room_ids_int: Optional[torch.Tensor]
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    draft_probs: Optional[torch.Tensor]
    hidden_states: Optional[torch.Tensor]
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]
    dsa_seed_topk: Optional[torch.Tensor] = None


class EAGLEDraftCudaGraphRunner(DecodeCudaGraphRunner):
    """EAGLE draft cuda-graph runner.

    Subclasses DecodeCudaGraphRunner to inherit the outer capture
    loop (capture()), bucket-padding helper (_pad_to_bucket),
    and the backend-driven capture/replay scaffolding. EAGLE-specific
    bits — buffer dataclass, dummy ForwardBatch construction in
    capture_one_shape, replay output unwrap, and can_run_graph — are
    overridden.

    EAGLE does not call DecodeCudaGraphRunner.__init__ (that init
    sets up many decode-only fields like SWA/encoder-decoder/MLA-aware
    state). Instead it sets up its own state directly while making sure
    the parent's capture() / backend contract is satisfied.
    """

    def __init__(
        self,
        eagle_worker: EagleDraftWorker,
        *,
        draft_attn_backend=None,
        speculative_num_steps: Optional[int] = None,
    ):
        # Parse args
        self.eagle_worker = eagle_worker
        if not hasattr(eagle_worker, "model_runner"):
            # V2: EagleDraftWorker
            self.model_runner = model_runner = eagle_worker.draft_runner
        else:
            self.model_runner = model_runner = eagle_worker.model_runner

        # Fields the parent's capture() reads:
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.ps.tp_size
        self.attn_dp_size = model_runner.ps.attn_dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.enable_torch_compile = get_flags().capture.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.speculative_num_steps = (
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.draft_attn_backend = draft_attn_backend or model_runner.draft_attn_backend

        # Patch_model in parent's capture() needs an attn_backend reference.
        # EAGLE doesn't use it (capture_one_shape calls draft_forward instead),
        # but the field must exist.
        self.attn_backend = self.draft_attn_backend

        # Disable parent paths that don't apply to EAGLE.
        self.compile_bs = []  # disables patch_model torch.compile wrapping
        self.enable_pdmux = False
        self.record_nolora_graph = False
        self.is_dllm = False

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        # Capture-time globals required by parent's capture_one_shape signature.
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.LAST

        # Bucket sizes
        self.capture_bs, _ = get_batch_sizes_to_capture(model_runner)
        # Static capture width.
        self.captured_req_width = resolve_num_tokens_per_req(
            phase="draft_decode", server_args=model_runner.server_args
        )
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.captured_req_width

        # Attention backend init
        self.draft_attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)
        self.seq_len_fill_value = self.draft_attn_backend.attn_backends[
            0
        ].get_cuda_graph_seq_len_fill_value()
        self.extend_seq_lens_cpu = [self.seq_len_fill_value] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Static buffers
        with torch.device(model_runner.device):
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            out_cache_loc = torch.zeros(
                (self.max_num_token * self.speculative_num_steps,),
                dtype=self._cache_loc_dtype(),
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
            rids_int = (
                torch.zeros((self.max_bs,), dtype=torch.int64)
                if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get()
                else None
            )
            bootstrap_room_ids_int = (
                torch.full((self.max_bs,), -1, dtype=torch.int64)
                if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get()
                else None
            )
            seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
            )
            extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)
            topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)
            draft_probs = (
                torch.zeros(
                    (self.max_bs, self.model_runner.model_config.vocab_size),
                    dtype=torch.float32,
                )
                if self.model_runner.server_args.speculative_use_rejection_sampling
                else None
            )
            _hidden_size, _hidden_dtype = get_draft_recurrent_hidden_state_spec(
                model_runner
            )
            hidden_states = (
                torch.zeros(
                    (self.max_bs, _hidden_size),
                    dtype=_hidden_dtype,
                )
                if _hidden_size is not None
                else None
            )

            self.temperatures = torch.ones((self.max_bs, 1), dtype=torch.float)

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (self.attn_dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.attn_dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

        seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64, device="cpu"
        )

        dsa_seed_topk = (
            torch.zeros(
                (self.max_bs, self.eagle_worker.dsa_index_topk),
                dtype=torch.int32,
                device=model_runner.device,
            )
            if self.eagle_worker.seed_dsa_topk_from_draft_extend
            else None
        )

        self.buffers = EagleDraftInputBuffers(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=draft_probs,
            hidden_states=hidden_states,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            dsa_seed_topk=dsa_seed_topk,
        )
        self.buffers.share_buffers()

        self.backend = resolve_decode_backend(self)

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _replay_graph(self, shape_key, forward_batch):
        return self.backend.replay(shape_key, forward_batch)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _cache_loc_dtype(self):
        return torch.int64

    def _make_graph_key(self, bs, stream_idx=None, variant_label=None):
        # EAGLE doesn't use stream_idx / lora variants.
        return ShapeKey(size=bs)

    # -----------------------------------------------------------------
    # can_run_graph
    # -----------------------------------------------------------------
    def can_run_graph(self, forward_batch: ForwardBatch):
        # Uniform-width replay invariant: the batch's actual per-request width
        # must match this runner's capture width; anything else falls back to
        # eager. (Unset widths pass: not every path fills the field yet.)
        spec_info = forward_batch.spec_info
        if (
            spec_info is not None
            and spec_info.num_tokens_per_req > 0
            and spec_info.num_tokens_per_req != self.captured_req_width
        ):
            return False

        if self.require_mlp_tp_gather:
            # Raw sync values are per-rank request counts on decode-family rounds.
            cuda_graph_bs = max(forward_batch.original_global_num_tokens_cpu)
        else:
            cuda_graph_bs = forward_batch.batch_size

        is_bs_supported = (
            self.backend.can_run(forward_batch, cuda_graph_bs)
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    # -----------------------------------------------------------------
    # Capture (per-shape)
    # -----------------------------------------------------------------
    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        num_seqs = size  # EAGLE legacy name
        buffers = self.buffers
        num_tokens = num_seqs * self.captured_req_width

        # Graph inputs
        req_pool_indices = buffers.req_pool_indices[:num_seqs]
        seq_lens = buffers.seq_lens[:num_seqs]
        seq_lens_cpu = buffers.seq_lens_cpu[:num_seqs]
        extend_seq_lens = buffers.extend_seq_lens[:num_seqs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:num_seqs]
        out_cache_loc = buffers.out_cache_loc[: num_tokens * self.speculative_num_steps]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        rids_int = buffers.rids_int[:num_seqs] if buffers.rids_int is not None else None
        bootstrap_room_ids_int = (
            buffers.bootstrap_room_ids_int[:num_seqs]
            if buffers.bootstrap_room_ids_int is not None
            else None
        )
        hidden_states = (
            buffers.hidden_states[:num_seqs]
            if buffers.hidden_states is not None
            else None
        )
        topk_p = buffers.topk_p[:num_seqs]
        topk_index = buffers.topk_index[:num_seqs]
        draft_probs = (
            buffers.draft_probs[:num_seqs] if buffers.draft_probs is not None else None
        )

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [num_tokens] * self.attn_dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu,
                dtype=torch.int32,
                device=buffers.input_ids.device,
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
        else:
            global_dp_buffer_len = None
            global_num_tokens = None
            global_num_tokens_for_logprob = None

        capture_mode = (
            CaptureHiddenMode.NULL
            if self.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        spec_info = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=draft_probs,
            hidden_states=hidden_states,
            capture_hidden_mode=capture_mode,
        )
        if self.buffers.dsa_seed_topk is not None:
            spec_info.dsa_topk_indices = self.buffers.dsa_seed_topk[:num_seqs]

        sampling_info = SamplingBatchInfo(
            temperatures=self.temperatures[:num_seqs],
            top_ps=torch.ones((num_seqs,), dtype=torch.float),
            top_ks=torch.full((num_seqs,), -1, dtype=torch.int32),
            min_ps=torch.zeros((num_seqs,), dtype=torch.float),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=self.model_runner.model_config.vocab_size,
        )

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=num_seqs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            sampling_info=sampling_info,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
            capture_hidden_mode=(
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            ),
        )

        def run_once():
            self.draft_attn_backend.init_forward_metadata_in_graph(forward_batch)

            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states
            dsa_topk_indices_backup = forward_batch.spec_info.dsa_topk_indices

            ret = self.eagle_worker.draft_forward(forward_batch)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            forward_batch.spec_info.dsa_topk_indices = dsa_topk_indices_backup
            forward_batch.positions.sub_(self.eagle_worker.speculative_num_steps - 1)
            return ret

        with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):
            self.draft_attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=True
            )
            # The capture batch is planned here (out-of-forward), so the
            # per-step forwards inside draft_forward must not re-plan.
            forward_batch.mark_forward_metadata_ready()
            self.deepep_adapter.capture(is_extend_in_batch=False)
            shape_key = self._make_graph_key(num_seqs)
            post_warmup_hook = getattr(
                self.draft_attn_backend, "on_after_cuda_graph_warmup", None
            )
            maybe_flashinfer_autotune_speculative_draft(
                self,
                run_once,
                post_warmup_hook=post_warmup_hook,
                skip_logits=False,
            )
            self.backend.capture_one(
                shape_key,
                run_once,
                dummies=None,
                post_warmup_hook=post_warmup_hook,
            )

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        parent_list, top_scores_index, draft_tokens, draft_probs = (
            t[:raw_bs] if t is not None else None for t in out
        )
        return parent_list, top_scores_index, draft_tokens, draft_probs

    # -----------------------------------------------------------------
    # Replay
    # -----------------------------------------------------------------
    def execute(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        self.deepep_adapter.replay()
        buffers = self.buffers

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.captured_req_width

        # Pad to nearest captured shape
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.captured_req_width
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max_num_tokens
            )
            bs = self._pad_to_bucket(int(max_batch_size), self.capture_bs)
        else:
            bs = self._pad_to_bucket(raw_bs, self.capture_bs)

        if bs != raw_bs:
            buffers.seq_lens.fill_(self.seq_len_fill_value)
            buffers.out_cache_loc.zero_()
            buffers.positions.zero_()
            if buffers.rids_int is not None:
                buffers.rids_int.zero_()
            if buffers.bootstrap_room_ids_int is not None:
                buffers.bootstrap_room_ids_int.fill_(-1)
            buffers.topk_p.zero_()
            buffers.topk_index.zero_()
            if buffers.draft_probs is not None:
                buffers.draft_probs.zero_()
            if buffers.hidden_states is not None:
                buffers.hidden_states.zero_()
            if buffers.dsa_seed_topk is not None:
                buffers.dsa_seed_topk.zero_()
            buffers.req_pool_indices.zero_()

        num_tokens = bs * self.captured_req_width

        maybe_detect_nan(
            forward_batch.spec_info.topk_p,
            "EagleDraftCudaGraphRunner.replay: topk_p",
        )
        maybe_detect_oob(
            forward_batch.spec_info.topk_index,
            0,
            self.model_runner.model_config.vocab_size,
            "EagleDraftCudaGraphRunner.replay: topk_index vs vocab_size="
            f"{self.model_runner.model_config.vocab_size}",
        )

        # Common inputs — batch the small per-field device copies into a grouped
        # foreach copy (one foreach call per dtype pair) to cut launch overhead.
        # hidden_states is handled separately below (see note), and seq_lens_cpu
        # is handled further down since it lives on host.
        copy_dsts = [
            buffers.seq_lens[:raw_bs],
            buffers.out_cache_loc[: raw_num_token * self.speculative_num_steps],
            buffers.positions[:raw_num_token],
            buffers.topk_p[:raw_bs],
            buffers.topk_index[:raw_bs],
            buffers.req_pool_indices[:raw_bs],
        ]
        copy_srcs = [
            forward_batch.seq_lens,
            forward_batch.out_cache_loc,
            forward_batch.positions,
            forward_batch.spec_info.topk_p,
            forward_batch.spec_info.topk_index,
            forward_batch.req_pool_indices,
        ]
        if buffers.rids_int is not None and forward_batch.rids_int is not None:
            copy_dsts.append(buffers.rids_int[:raw_bs])
            copy_srcs.append(forward_batch.rids_int)
        if (
            buffers.bootstrap_room_ids_int is not None
            and forward_batch.bootstrap_room_ids_int is not None
        ):
            copy_dsts.append(buffers.bootstrap_room_ids_int[:raw_bs])
            copy_srcs.append(forward_batch.bootstrap_room_ids_int)
        _grouped_foreach_copy_(copy_dsts, copy_srcs)

        # hidden_states is large + contiguous: copy_() uses the cudaMemcpyAsync
        # DMA engine; foreach would force the ~3x slower compute-kernel copy.
        if (
            buffers.draft_probs is not None
            and forward_batch.spec_info.draft_probs is not None
        ):
            buffers.draft_probs[:raw_bs].copy_(forward_batch.spec_info.draft_probs)
        if (
            buffers.hidden_states is not None
            and forward_batch.spec_info.hidden_states is not None
        ):
            buffers.hidden_states[:raw_bs].copy_(forward_batch.spec_info.hidden_states)
        if buffers.dsa_seed_topk is not None:
            seed = forward_batch.spec_info.dsa_topk_indices
            if seed is not None:
                buffers.dsa_seed_topk[:raw_bs].copy_(seed)
            else:
                buffers.dsa_seed_topk[:raw_bs].zero_()
        # Only rejection sampling reads temperatures (renorm_draft_probs); skip
        # the copy otherwise to keep the non-RS path free of extra work.
        if (
            self.model_runner.server_args.speculative_use_rejection_sampling
            and forward_batch.sampling_info is not None
        ):
            self.temperatures[:raw_bs].copy_(
                forward_batch.sampling_info.temperatures[:raw_bs]
            )

        # TODO(ch-wan): support num_token_non_padded
        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(bs * self.captured_req_width)
            buffers.global_num_tokens_for_logprob_gpu.fill_(
                bs * self.captured_req_width
            )

        # Save the raw seq_lens_sum; it is restored after replay. While the graph
        # runs it must reflect the padded fake rows (set below), since draft decode
        # backends read seq_lens_sum to size/slice kv_indices.
        raw_seq_lens_sum = forward_batch.seq_lens_sum

        if bs != raw_bs:
            forward_batch.batch_size = bs
            forward_batch.seq_lens = buffers.seq_lens[:bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:bs]
            forward_batch.positions = buffers.positions[:num_tokens]
            if raw_seq_lens_sum is not None:
                forward_batch.seq_lens_sum = (
                    raw_seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value
                )
            if buffers.rids_int is not None and forward_batch.rids_int is not None:
                forward_batch.rids_int = buffers.rids_int[:bs]
            if (
                buffers.bootstrap_room_ids_int is not None
                and forward_batch.bootstrap_room_ids_int is not None
            ):
                forward_batch.bootstrap_room_ids_int = buffers.bootstrap_room_ids_int[
                    :bs
                ]

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
            forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:bs]

        # forward_batch.batch_size was overwritten to bs above when padding.
        self.draft_attn_backend.init_forward_metadata_out_graph(forward_batch)
        self.raw_bs = raw_bs
        self.bs = bs

        # Replay via backend
        shape_key = self._make_graph_key(bs)
        timer_ctx = (
            self.model_runner.device_timer.wrap(metadata={"category": "eagle_draft"})
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with timer_ctx:
            out = self._replay_graph(shape_key, forward_batch)
        if self.buffers.dsa_seed_topk is not None:
            forward_batch.spec_info.dsa_topk_indices = None

        if bs != raw_bs:
            out = self._postprocess_output_to_raw_bs(out, raw_bs)
            forward_batch.batch_size = raw_bs
            forward_batch.positions = buffers.positions[:raw_num_token]
            forward_batch.seq_lens = buffers.seq_lens[:raw_bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:raw_bs]
            if buffers.rids_int is not None and forward_batch.rids_int is not None:
                forward_batch.rids_int = buffers.rids_int[:raw_bs]
            if (
                buffers.bootstrap_room_ids_int is not None
                and forward_batch.bootstrap_room_ids_int is not None
            ):
                forward_batch.bootstrap_room_ids_int = buffers.bootstrap_room_ids_int[
                    :raw_bs
                ]
            if forward_batch.seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:raw_bs]
            forward_batch.seq_lens_sum = raw_seq_lens_sum

        return out
