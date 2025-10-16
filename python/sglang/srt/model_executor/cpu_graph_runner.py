# Copyright 2023-2024 SGLang Team
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
"""Run the model with cpu torch compile."""

# The implementation of CPUGraphRunner follows the CudaGraphRunner

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union

import psutil
import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                dynamic=False,
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.freezing = True
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024
    monkey_patch_torch_compile()


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    server_args = model_runner.server_args
    # cpu torch compile only speeds up decoding by
    # reducing python overhead when bs is small
    capture_bs = list(range(1, 17))
    capture_bs = [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
    capture_bs = [bs for bs in capture_bs if bs <= model_runner.req_to_token_pool.size]
    capture_bs = list(sorted(set(capture_bs)))
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    return capture_bs


def register_fake_ops():
    """
    Registers fake/meta implementations for all custom sgl_kernel CPU operators
    using torch.library.register_fake to support torch.compile
    """

    none_return_ops = [
        "shm_allreduce",
        "bmm_cpu",
        "fused_add_rmsnorm_cpu",
        "decode_attention_cpu",
        "extend_attention_cpu",
    ]
    for op in none_return_ops:

        @torch.library.register_fake(f"sgl_kernel::{op}")
        def _(*args, **kwargs):
            return

    for op in [
        "rmsnorm_cpu",
        "l2norm_cpu",
        "fused_experts_cpu",
        "shared_expert_cpu",
    ]:

        @torch.library.register_fake(f"sgl_kernel::{op}")
        def _(input, *args, **kwargs):
            return torch.empty_like(input)

    @torch.library.register_fake("sgl_kernel::qkv_proj_with_rope")
    def _(
        hidden_states,
        q_a_proj_weight,
        q_b_proj_weight,
        kv_a_proj_weight,
        w_kc,
        q_a_layernorm_weight,
        kv_a_layernorm_weight,
        positions,
        cos_sin_cache,
        eps,
        use_int8_w8a8,
        use_fp8_w8a16,
        q_a_proj_scale,
        q_b_proj_scale,
        kv_a_proj_scale,
        is_vnni,
        block_size,
    ):
        num_seqs = hidden_states.shape[0]
        num_heads = w_kc.shape[0]
        kv_lora_rank = w_kc.shape[1]
        qk_rope_head_dim = kv_a_proj_weight.shape[0] - kv_lora_rank
        q_input = torch.empty(
            num_seqs,
            num_heads,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        k_input = torch.empty(
            num_seqs,
            1,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        v_input = k_input.narrow(-1, 0, kv_lora_rank)
        return q_input, k_input, v_input

    @torch.library.register_fake("sgl_kernel::rotary_embedding_cpu")
    def _(positions, query, key, head_size, cos_sin_cache, is_neox):
        if query.ndim == 2:
            return query, key
        else:
            return torch.empty_like(query), torch.empty_like(key)

    @torch.library.register_fake("sgl_kernel::qkv_proj_with_rope_fused_weight")
    def _(
        hidden_states,
        q_a_proj_weight,
        q_b_proj_weight,
        w_kc,
        q_a_layernorm_weight,
        kv_a_layernorm_weight,
        positions,
        cos_sin_cache,
        eps,
        use_int8_w8a8,
        use_fp8_w8a16,
        qkv_a_proj_scale,
        q_b_proj_scale,
        is_vnni,
        block_size,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
    ):
        num_seqs = hidden_states.shape[0]
        num_heads = w_kc.shape[0]
        kv_lora_rank = w_kc.shape[1]
        weight_chunks = torch.split(
            q_a_proj_weight, [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=0
        )
        qk_rope_head_dim = weight_chunks[1].shape[0] - kv_lora_rank
        q_input = torch.empty(
            num_seqs,
            num_heads,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        k_input = torch.empty(
            num_seqs,
            1,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        v_input = k_input.narrow(-1, 0, kv_lora_rank)
        return q_input, k_input, v_input

    @torch.library.register_fake("sgl_kernel::weight_packed_linear")
    def _(x, weight, bias, is_vnni):
        return x.new_empty(x.shape[0], weight.shape[0])

    @torch.library.register_fake("sgl_kernel::per_token_quant_int8_cpu")
    def _(input):
        M = input.shape[0]
        K = input.shape[1]
        Aq = input.new_empty(M, K, dtype=torch.int8)
        As = input.new_empty(M, dtype=torch.float32)
        return Aq, As

    @torch.library.register_fake("sgl_kernel::int8_scaled_mm_cpu")
    def _(mat1, mat2, scales1, scales2, bias, out_dtype, is_vnni):
        M = mat1.shape[0]
        N = mat2.shape[0]
        out = mat1.new_empty(M, N, dtype=out_dtype)
        return out

    @torch.library.register_fake("sgl_kernel::grouped_topk_cpu")
    def _(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    ):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        device = hidden_states.device
        topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
        topk_ids = torch.empty(shape, device=device, dtype=torch.int)
        return topk_weights, topk_ids

    @torch.library.register_fake("sgl_kernel::biased_grouped_topk_cpu")
    def _(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    ):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        device = hidden_states.device
        topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
        topk_ids = torch.empty(shape, device=device, dtype=torch.int)
        return topk_weights, topk_ids

    @torch.library.register_fake("sgl_kernel::topk_sigmoid_cpu")
    def _(hidden_states, gating_output, topk, renormalize):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        return (
            torch.empty(shape, device=hidden_states.device, dtype=torch.float),
            torch.empty(shape, device=hidden_states.device, dtype=torch.int),
        )

    @torch.library.register_fake("sgl_kernel::topk_softmax_cpu")
    def _(
        hidden_states,
        gating_output,
        topk,
        renormalize,
    ):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        return (
            torch.empty(shape, device=hidden_states.device, dtype=torch.float),
            torch.empty(shape, device=hidden_states.device, dtype=torch.int),
        )

    @torch.library.register_fake("sgl_kernel::silu_and_mul_cpu")
    def _(input):
        return input.new_empty(input.shape[0], input.shape[1] // 2)

    @torch.library.register_fake("sgl_kernel::int8_scaled_mm_with_quant")
    def _(
        mat1,
        mat2,
        scales2,
        bias,
        out_dtype,
        is_vnni,
    ):
        M = mat1.shape[0]
        N = mat2.shape[0]
        return mat1.new_empty(M, N, dtype=out_dtype)

    @torch.library.register_fake("sgl_kernel::fp8_scaled_mm_cpu")
    def _(
        mat1,
        mat2,
        scales2,
        block_size,
        bias,
        out_dtype,
        is_vnni,
    ):
        M = mat1.shape[0]
        N = mat2.shape[0]
        return mat1.new_empty(M, N, dtype=out_dtype)


# TODO Remove unnecessary settings for CPUGraphRunner.
# Re-abstract the graph runner and restructure CPUGraphRunner to reuse the same logic.
class CPUGraphRunner:
    """A CPUGraphRunner runs the forward pass of a model with cpu torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        assert (
            not self.model_runner.server_args.enable_lora
        ), "CPUGraphRunner does not support LoRA yet."
        assert (
            not self.enable_two_batch_overlap
        ), "CPUGraphRunner does not support two batch overlap yet."
        assert (
            not self.require_mlp_tp_gather
        ), "CPUGraphRunner does not support MLP TP gather yet."
        assert (
            not self.require_mlp_sync
        ), "CPUGraphRunner does not support MLP sync yet."
        assert (
            not self.require_gathered_buffer
        ), "CPUGraphRunner does not support gathered buffer yet."
        assert (
            model_runner.spec_algorithm == SpeculativeAlgorithm.NONE
        ), "CPUGraphRunner does not support speculative inference yet."
        # TODO add compile support for encoder-decoder models
        assert (
            not self.is_encoder_decoder
        ), "CPUGraphRunner does not support encoder-decoder models yet."
        assert self.dp_size == 1, "CPUGraphRunner does not support DP yet."
        assert self.pp_size == 1, "CPUGraphRunner does not support PP yet."

        # Batch sizes to capture
        self.capture_bs = get_batch_sizes_to_capture(model_runner)
        log_info_on_rank0(logger, f"Capture cpu graph bs {self.capture_bs}")
        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_graph_seq_len_fill_value()
        )

        if self.enable_torch_compile:
            register_fake_ops()
            set_torch_compile_config()

        # Graph inputs
        with torch.device(self.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int64)
            self.custom_mask = torch.ones(
                (
                    (self.seq_lens.sum().item() + self.max_num_token)
                    * self.num_tokens_per_bs
                ),
                dtype=torch.bool,
                device=self.device,
            )

        # Capture
        try:
            self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture CPU graph failed: {e}\n{CPU_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        is_bs_supported = forward_batch.batch_size in self.graphs

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )

        return is_bs_supported and capture_hidden_mode_matches

    def capture(self) -> None:
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.capture_bs)
        )
        for bs in capture_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = psutil.virtual_memory().available / (1 << 30)
                capture_range.set_description(
                    f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                )

            with patch_model(
                self.model_runner.model,
                bs in self.capture_bs,
                num_tokens=bs * self.num_tokens_per_bs,
                tp_group=self.model_runner.tp_group,
            ) as forward:
                (
                    graph,
                    output_buffers,
                ) = self.capture_one_batch_size(bs, forward)
                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers

    def capture_one_batch_size(self, bs: int, forward: Callable):
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        mrope_positions = self.mrope_positions[:, :bs]
        self.num_token_non_padded[...] = num_tokens

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=self.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
        )

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        # Do infernence to avoid setting attr at runtime, e.g.,
        # self.attn_mha.kv_b_proj = self.kv_b_proj for full graph compile on CPU
        self.model_runner.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            logits_output_or_pp_proxy_tensors = forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
            )
            return logits_output_or_pp_proxy_tensors

        with torch.no_grad():
            for _ in range(2):
                self.model_runner.tp_group.barrier()
                out = run_once()
            return forward, out

    def recapture_if_needed(self, forward_batch: ForwardBatch):

        # If the required capture_hidden_mode changes, we need to recapture the graph

        # These are the different factors that can influence the capture_hidden_mode
        capture_hidden_mode_required_by_forward_batch = (
            forward_batch.capture_hidden_mode
        )
        capture_hidden_mode_required_by_spec_info = getattr(
            forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
        )
        capture_hidden_mode_required_for_returning_hidden_states = (
            CaptureHiddenMode.FULL
            if self.model_runner.server_args.enable_return_hidden_states
            else CaptureHiddenMode.NULL
        )

        # Determine the highest capture_hidden_mode required
        # (If we have FULL, we can emulate LAST or NULL)
        # (If we have LAST, we can emulate NULL)
        required_capture_hidden_mode = max(
            capture_hidden_mode_required_by_forward_batch,
            capture_hidden_mode_required_by_spec_info,
            capture_hidden_mode_required_for_returning_hidden_states,
        )

        # If the current hidden mode is no longer aligned with the required hidden mode, we need to set it to what is required and re-capture
        if self.capture_hidden_mode != required_capture_hidden_mode:
            self.capture_hidden_mode = required_capture_hidden_mode
            self.capture()

    # TODO add padding support for CPUGraphRunner
    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        assert (
            pp_proxy_tensors is None
        ), "PPProxyTensors is not supported in CPUGraphRunner yet."
        self.recapture_if_needed(forward_batch)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        output = self.graphs[forward_batch.batch_size](
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )
        return output

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if self.model_runner.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.custom_mask,
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )

        return spec_info


CPU_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --torch-compile-max-bs to a smaller value (e.g., 8)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
