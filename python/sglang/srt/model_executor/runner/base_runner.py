# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class shared by EagerRunner and BaseCudaGraphRunner."""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    NgramEmbeddingInfo,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    run_flashinfer_autotune_forward,
    should_run_flashinfer_autotune,
)
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.speculative.spec_info import create_dummy_verify_input
from sglang.srt.utils import (
    empty_context,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _allocate_decode_buffers(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    dp_size: int,
    pp_size: int,
    is_encoder_decoder: bool,
    require_mlp_tp_gather: bool,
    seq_len_fill_value: int,
    encoder_len_fill_value: int,
    num_tokens_per_req: int,
    cache_loc_dtype: torch.dtype,
    enable_mamba_track: bool,
    ne_token_table: Optional[torch.Tensor] = None,
    hc_hidden_size: Optional[int] = None,
    pp_proxy_topk_size: Optional[int] = None,
) -> SimpleNamespace:
    """Allocate the FB-shared decode buffers."""
    with torch.device(device):
        input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
        input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
        req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
        seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int64)
        out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
        positions = torch.zeros((max_num_token,), dtype=torch.int64)
        mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
        num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
        custom_mask = torch.ones(
            (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_req,
            dtype=torch.bool,
        )
        next_token_logits_buffer = torch.zeros(
            (max_num_token, vocab_size),
            dtype=torch.float,
        )
        mamba_track_indices = (
            torch.zeros((max_bs,), dtype=torch.int64) if enable_mamba_track else None
        )
        mamba_track_mask = (
            torch.zeros((max_bs,), dtype=torch.bool) if enable_mamba_track else None
        )

        if pp_size > 1:
            # mHC (e.g. DSV4) flattens residual into hidden_states (size = hc_hidden_size).
            is_mhc = hc_hidden_size is not None
            hs = hc_hidden_size if is_mhc else hidden_size
            pp_proxy_tensors = {
                "hidden_states": torch.zeros((max_bs, hs), dtype=dtype),
            }
            if not is_mhc:
                pp_proxy_tensors["residual"] = torch.zeros(
                    (max_bs, hidden_size), dtype=dtype
                )
            if pp_proxy_topk_size is not None:
                pp_proxy_tensors["topk_indices"] = torch.zeros(
                    (max_num_token, pp_proxy_topk_size), dtype=torch.int32
                )
        else:
            pp_proxy_tensors = None

        if is_encoder_decoder:
            encoder_lens = torch.full(
                (max_bs,), encoder_len_fill_value, dtype=torch.int32
            )
        else:
            encoder_lens = None

        if require_mlp_tp_gather:
            global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
            global_num_tokens_for_logprob_gpu = torch.zeros(
                (dp_size,), dtype=torch.int32
            )
        else:
            global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
            global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)

        ngram_embedding_info = (
            NgramEmbeddingInfo(
                token_table=ne_token_table,
                column_starts=torch.zeros([max_bs], dtype=torch.int32),
                req_lens=torch.ones([max_bs], dtype=torch.int32),
                out_column_starts=torch.zeros([max_bs], dtype=torch.int32),
                out_req_lens=torch.ones([max_bs], dtype=torch.int32),
                skip_token_table_update=torch.zeros([max_bs], dtype=torch.bool),
            )
            if ne_token_table is not None
            else None
        )

        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            rids_int = torch.zeros((max_bs,), dtype=torch.int64)
            bootstrap_room_ids_int = torch.full((max_bs,), -1, dtype=torch.int64)
        else:
            rids_int = None
            bootstrap_room_ids_int = None

    seq_lens_cpu = torch.full(
        (max_bs,),
        seq_len_fill_value,
        dtype=torch.int64,
        device="cpu",
    )

    return SimpleNamespace(
        input_ids=input_ids,
        input_embeds=input_embeds,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        out_cache_loc=out_cache_loc,
        positions=positions,
        mrope_positions=mrope_positions,
        num_token_non_padded=num_token_non_padded,
        custom_mask=custom_mask,
        next_token_logits_buffer=next_token_logits_buffer,
        mamba_track_indices=mamba_track_indices,
        mamba_track_mask=mamba_track_mask,
        encoder_lens=encoder_lens,
        global_num_tokens_gpu=global_num_tokens_gpu,
        global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        pp_proxy_tensors=pp_proxy_tensors,
        ngram_embedding_info=ngram_embedding_info,
        rids_int=rids_int,
        bootstrap_room_ids_int=bootstrap_room_ids_int,
    )


class BaseRunner(ABC):
    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.enable_pdmux = model_runner.server_args.enable_pdmux
        self.enable_return_hidden_states = (
            model_runner.server_args.enable_return_hidden_states
        )
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.tbo_plugin = TboCudaGraphRunnerPlugin()

    def warmup(self) -> None:
        """Run kernel warmup + autotune once, gated by mr._kernel_warmed_up."""
        mr = self.model_runner
        if getattr(mr, "_kernel_warmed_up", False):
            return
        mr._kernel_warmed_up = True

        if mr.device != "cuda":
            return

        self._pre_initialize_flashinfer_allreduce_workspace()
        self._pre_initialize_fi_a2a_workspace()

        if should_run_flashinfer_autotune(self.model_runner):
            buffers, batch_size = self._autotune_buffers()
            assert (
                buffers is not None
            ), "_autotune_buffers() must return a reusable buffer set for autotune"
            self._flashinfer_autotune(buffers=buffers, batch_size=batch_size)

        if (
            envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.get()
            and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and mr.ps.pp_size > 1
            and not mr.spec_algorithm.is_speculative()
        ):
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                pp_parallel_deep_gemm_warmup,
            )

            pp_parallel_deep_gemm_warmup(self)

    def _pre_initialize_flashinfer_allreduce_workspace(self):
        """Allocate flashinfer allreduce workspaces; must run before CG capture
        to keep broadcasts/barriers outside the capture context (else deadlock
        with custom_all_reduce.register_graph_buffers).
        """
        mr = self.model_runner
        if mr.server_args.flashinfer_allreduce_fusion_backend is None:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import pre_initialize_workspaces

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=mr.model_config.hidden_size,
            dtype=mr.dtype,
        )

    def _pre_initialize_fi_a2a_workspace(self):
        """Allocate the FlashInfer MNNVL all-to-all workspace for the fi_a2a DCP
        comm backend; must run before CG capture (it syncs the stream + barriers
        cross-rank, uncapturable) and raises early on non-MNNVL platforms.
        """
        mr = self.model_runner
        if mr.server_args.dcp_size <= 1 or mr.server_args.dcp_comm_backend != "fi_a2a":
            return

        from sglang.srt.layers.dcp import init_fi_a2a_workspace

        init_fi_a2a_workspace(get_parallel().dcp_group)

    def _flashinfer_autotune(self, *, buffers, batch_size):
        """Run flashinfer autotune.

        buffers / batch_size: a prepared static decode-buffer set and its bs,
        reused for the dummy forward instead of allocating a throwaway set.
        Supplied by warmup() (the decode runner's captured buffers when a graph
        runner exists; a freshly-allocated dummy set in the eager path).
        """
        mr = self.model_runner
        canary_run_ctx = (
            c.with_active_single_forward_manager(0)
            if (c := mr.canary_manager) is not None
            else empty_context()
        )

        def forward_fn():
            self._dummy_run(
                batch_size=batch_size,
                buffers=buffers,
                run_ctx=canary_run_ctx,
            )

        run_flashinfer_autotune_forward(self.model_runner, forward_fn, skip_logits=True)

    def _alloc_dummy_decode_buffers(self, max_bs: int, *, num_tokens_per_req: int = 1):
        """Allocate one static decode-buffer set for a dummy forward, sized to
        (max_bs, max_bs * num_tokens_per_req).

        The PP-parallel DeepGEMM warmup sweeps batch sizes far larger than any
        runner's max_bs (up to ~n_sms*block_m), so no pre-allocated runner buffer
        set fits; it builds one here and hands it to _dummy_run (reused across the
        sweep; _dummy_run slices it per shape). Eager FlashInfer autotune also
        allocates decode-shaped scratch buffers here. Decode cuda-graph autotune
        reuses the captured runner buffers instead.
        """
        mr = self.model_runner
        return _allocate_decode_buffers(
            device=mr.device,
            max_bs=max_bs,
            max_num_token=max_bs * num_tokens_per_req,
            hidden_size=mr.model_config.hidden_size,
            vocab_size=mr.model_config.vocab_size,
            dtype=mr.model_config.dtype,
            dp_size=mr.server_args.dp_size,
            pp_size=mr.server_args.pp_size,
            is_encoder_decoder=mr.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather(mr.server_args),
            seq_len_fill_value=mr.attn_backend.get_cuda_graph_seq_len_fill_value(),
            encoder_len_fill_value=(
                getattr(mr.model_config.hf_config, "max_source_positions", 0)
                if mr.model_config.is_encoder_decoder
                else 0
            ),
            num_tokens_per_req=num_tokens_per_req,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            ne_token_table=(
                mr.ngram_embedding_manager.table
                if mr.ngram_embedding_manager.enabled
                else None
            ),
            hc_hidden_size=getattr(mr.model_config, "hc_hidden_size", None),
            pp_proxy_topk_size=mr.get_pp_proxy_topk_size(),
        )

    def _dummy_run(
        self,
        batch_size: int,
        run_ctx=None,
        forward_mode_override: Optional[ForwardMode] = None,
        *,
        buffers,
    ):
        """Run a dummy forward pass for warmup/profiling.

        forward_mode_override forces EXTEND/DECODE regardless of
        is_generation (used by the PP-parallel DeepGEMM warmup).

        buffers: a prepared static buffer set (or lightweight adapter exposing
        the same fields), sized >= this dummy shape, which _dummy_run slices to
        (batch_size, num_tokens). The caller owns the shape and the allocation --
        the flashinfer autotune reuses an existing runner's buffers via
        _autotune_buffers (the eager input registry, or the decode cuda-graph
        runner's captured buffers); the PP-DeepGEMM warmup builds one via
        _alloc_dummy_decode_buffers. _dummy_run never allocates and never re-pads
        (autotune must run at the reused shape; the PP warmup pre-pads and sizes
        its buffer to match). next_token_logits_buffer is optional -- a live
        autotune forward returns logits fresh, so the eager-reuse path passes
        None (only the PP warmup set still carries one).
        """
        mr = self.model_runner
        if forward_mode_override is not None:
            capture_forward_mode = forward_mode_override
        elif mr.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_req = 1
        if mr.spec_algorithm.is_speculative():
            if mr.is_draft_worker:
                assert (
                    mr.spec_algorithm.supports_target_verify_for_draft()
                ), "This should not happen"
            capture_forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens_per_req = mr.decode_num_tokens_per_req()

        if mr.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_req

        # Caller owns the shape: passes a static buffer >= the dummy shape; no
        # allocation, no re-padding (would overflow the reused buffers).
        assert (
            buffers is not None
            and num_tokens <= buffers.input_ids.shape[0]
            and batch_size <= buffers.seq_lens.shape[0]
        ), (
            f"_dummy_run needs a static buffer >= (num_tokens={num_tokens}, "
            f"batch_size={batch_size}); got "
            + (
                "None"
                if buffers is None
                else f"(input_ids={buffers.input_ids.shape[0]}, "
                f"seq_lens={buffers.seq_lens.shape[0]})"
            )
        )

        if get_flags().capture.enable_torch_compile:
            set_torch_compile_config()
            should_disable_torch_compile = not getattr(
                mr.model, "_can_torch_compile", True
            )
            if should_disable_torch_compile:
                log_info_on_rank0(
                    logger,
                    "Transformers backend model reports it is not torch.compile "
                    "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
                )
                get_flags().capture.enable_torch_compile = False

        # NOTE: aux hidden state capture (eagle3/dflash) is already
        # configured by init_aux_hidden_state_capture() in initialize().

        # Token-axis buffer views and counters.
        input_ids = buffers.input_ids[:num_tokens]
        positions = buffers.positions[:num_tokens]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        buffers.num_token_non_padded[...] = num_tokens

        # Batch-axis buffer views.
        req_pool_indices = buffers.req_pool_indices[:batch_size]
        seq_lens = buffers.seq_lens[:batch_size]
        seq_lens_cpu = buffers.seq_lens_cpu[:batch_size]

        # Optional buffer views.
        # Eager-reuse drops the logits buffer; only buffer sets that carry one slice it.
        next_token_logits_buffer = (
            buffers.next_token_logits_buffer[:num_tokens]
            if buffers.next_token_logits_buffer is not None
            else None
        )
        encoder_lens = (
            buffers.encoder_lens[:batch_size]
            if buffers.encoder_lens is not None
            else None
        )

        # For extend mode
        if capture_forward_mode == ForwardMode.EXTEND:
            seq_len_fill_value = mr.attn_backend.get_cuda_graph_seq_len_fill_value()
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=mr.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=mr.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_req, dtype=torch.int32, device=mr.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if mr.server_args.pp_size > 1:
            # PP0 already cp-split hidden_states before send.
            pp_hidden_tokens = num_tokens
            if (
                capture_forward_mode == ForwardMode.EXTEND
                and mr.ps.pp_rank != 0
                and mr.ps.attn_cp_size > 1
            ):
                pp_hidden_tokens = num_tokens // mr.ps.attn_cp_size
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:pp_hidden_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        # TP-gather requirements for global token metadata.
        require_mlp_tp_gather_ = require_mlp_tp_gather(mr.server_args)
        require_attn_tp_gather_ = require_attn_tp_gather(mr.server_args)
        if require_gathered_buffer(mr.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather_

        if require_mlp_tp_gather_:
            global_num_tokens_cpu = [num_tokens] * mr.server_args.dp_size
        elif require_attn_tp_gather_:
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=mr.device
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
        else:
            global_dp_buffer_len = None
            global_num_tokens_cpu = None

        # Speculative metadata and hidden-state capture mode.
        spec_info = create_dummy_verify_input(
            mr.spec_algorithm,
            mr.server_args,
            buffers.custom_mask,
            num_tokens_per_req,
            mr.is_draft_worker,
        )
        if spec_info is not None and (
            mr.spec_algorithm.is_eagle() or mr.spec_algorithm.is_standalone()
        ):
            # MTP models (e.g. deepseek_nextn) read spec_info.hidden_states
            # during forward; provide a dummy so warmup doesn't crash.
            spec_info.hidden_states = torch.zeros(
                (num_tokens, mr.model_config.hidden_size),
                dtype=mr.dtype,
                device=mr.device,
            )
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        # Optional LoRA metadata.
        if mr.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=mr.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if buffers.ngram_embedding_info is not None:
            forward_batch.ngram_embedding_info = buffers.ngram_embedding_info.slice(
                batch_size
            )
        if lora_ids is not None:
            mr.lora_manager.prepare_lora_batch(forward_batch)

        forward_batch = mr.prepare_dummy_forward_batch(forward_batch)
        mr.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            # Reused dummy batches may carry DP-local lazy caches from a prior
            # forward. Clear them, then refresh the process-wide DP buffer and
            # MoE mode metadata read by model code during this standalone run.
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                mr.server_args.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(mr.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not mr.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = mr.model.forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(mr.device).synchronize()
        mr.tp_group.barrier()
        with forward_context(ForwardContext(attn_backend=mr.attn_backend)):
            with torch.inference_mode(), run_ctx or empty_context():
                run_once()

    def _autotune_buffers(self) -> Tuple[Optional[Any], Optional[int]]:
        """Return (buffers, bs) for the autotune dummy forward to reuse; the
        EagerRunner and DecodeCudaGraphRunner override this."""
        return None, None

    @abstractmethod
    def can_run_graph(self, forward_batch: ForwardBatch) -> bool: ...

    @abstractmethod
    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def execute(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...
