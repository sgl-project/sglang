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

import bisect
import inspect
import logging
import os
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
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.graph_runner import GraphRunner
from sglang.srt.patch_torch import monkey_patch_torch_compile
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_available_gpu_memory, rank0_log

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

    # TODO add cpu_graph_bs to server_args
    # capture_bs = server_args.cuda_graph_bs

    # cpu torch compile only speeds up decoding by
    # reducing python overhead when bs is small
    capture_bs = list(range(1, 9)) + list(range(10, 17, 2)) + list([20, 24, 28, 32, 36, 40, 50, 64, 80, 100, 120, 140, 160, 180, 200, 220, 236, 240, 242, 244, 246, 248, 250, 256, 260, 280, 300, 320, 360])

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
        "gemma3_rmsnorm_cpu",
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

    @torch.library.register_fake("sgl_kernel::int4_scaled_mm_cpu_with_quant")
    def _(
        input,
        weight,
        weight_scales,
        weight_qzeros,
        compensation,
        bias,
        output_dtype,
    ):
        out_sizes = list(input.shape)
        N = weight.shape[0] * weight.shape[-1] * 2
        out_sizes[-1] = N
        return input.new_empty(out_sizes, dtype=output_dtype)

    @torch.library.register_fake("sgl_kernel::int4_scaled_mm_cpu")
    def _(
        x,
        w,
        w_zeros,
        w_scales,
        bias,
    ):
        M = x.shape[0]
        N = w.shape[0]
        return x.new_empty(M, N)

    @torch.library.register_fake("sgl_kernel::gelu_and_mul_cpu")
    def _(
        input,
    ):
        sizes = list(input.shape)
        sizes[-1] = sizes[-1] // 2
        return input.new_empty(sizes)

    @torch.library.register_fake("sgl_kernel::gelu_tanh_and_mul_cpu")
    def _(
        input,
    ):
        sizes = list(input.shape)
        sizes[-1] = sizes[-1] // 2
        return input.new_empty(sizes)

    @torch.library.register_fake("sgl_kernel::silu_and_mul_cpu")
    def _(
        input,
    ):
        sizes = list(input.shape)
        sizes[-1] = sizes[-1] // 2
        return input.new_empty(sizes)


class CPUGraphRunner(GraphRunner):
    """A CPUGraphRunner runs the forward pass of a model with cpu torch.compile."""

    def __init__(self, model_runner: ModelRunner):

        super().__init__(model_runner, device="cpu")
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
        rank0_log(f"Capture cpu graph bs {self.capture_bs}")
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
        cpu_graph_bs = forward_batch.batch_size

        is_bs_supported = (
            cpu_graph_bs in self.graphs
            if self.disable_padding
            else cpu_graph_bs <= self.max_bs
        )

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
        avail_mem = psutil.virtual_memory().available
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.capture_bs)
        )
        for bs in capture_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
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

        gathered_buffer = None

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
            gathered_buffer=gathered_buffer,
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

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)

        if pp_proxy_tensors:
            for key in self.pp_proxy_tensors.keys():
                dim = pp_proxy_tensors[key].shape[0]
                self.pp_proxy_tensors[key][:dim].copy_(pp_proxy_tensors[key])

        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_bs].copy_(forward_batch.mrope_positions)
        if enable_num_token_non_padded(self.model_runner.server_args):
            self.num_token_non_padded.copy_(forward_batch.num_token_non_padded)
        if forward_batch.forward_mode.is_idle() and forward_batch.spec_info is not None:
            forward_batch.spec_info.custom_mask = self.custom_mask

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        output = self.graphs[self.bs](
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )
        if isinstance(output, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                next_token_logits=output.next_token_logits[: self.raw_num_token],
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})


CPU_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --torch-compile-max-bs to a smaller value (e.g., 8)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
