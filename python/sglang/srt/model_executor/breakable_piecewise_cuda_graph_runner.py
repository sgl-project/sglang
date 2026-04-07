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
"""Piecewise CUDA graph runner using breakable CUDA graph.

Instead of torch.compile + FX graph splitting, this runner uses breakable CUDA
graph to insert graph breaks at attention layers. The model forward is captured
as a single breakable CUDA graph that automatically splits at @non_graph points
(radix attention for dense models).
"""

from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, Union

import torch
import tqdm

from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    set_forward_context,
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.dp_attention import (
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphContext,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.srt.model_executor.forward_batch_info import (
    PPProxyTensors,
)
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
    freeze_gc,
)
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

# Bridge buffers for reducing graph-owned persistent memory at break points.
# Pre-allocated OUTSIDE graph capture so they're not graph-owned.
# Reused across all layers (sequential execution) and all token sizes.
_bridge_buffers = None


def get_bridge_buffers():
    return _bridge_buffers


def set_bridge_buffers(buffers):
    global _bridge_buffers
    _bridge_buffers = buffers


class BridgeBuffers:
    """Pre-allocated tensors for q, k, v, output at graph break points.

    During graph capture, the model copies q/k/v into these buffers before
    the graph break. This makes the original tensors graph-private (freeable),
    while the bridge buffers persist at fixed addresses outside the graph.
    """

    def __init__(self, max_tokens: int, attention_layer, device, dtype):
        q_dim = attention_layer.tp_q_head_num * attention_layer.qk_head_dim
        k_dim = attention_layer.tp_k_head_num * attention_layer.qk_head_dim
        v_dim = attention_layer.tp_v_head_num * attention_layer.v_head_dim
        out_dim = attention_layer.tp_q_head_num * attention_layer.v_head_dim

        self.q = torch.empty((max_tokens, q_dim), dtype=dtype, device=device)
        self.k = torch.empty((max_tokens, k_dim), dtype=dtype, device=device)
        self.v = torch.empty((max_tokens, v_dim), dtype=dtype, device=device)
        self.output = torch.empty((max_tokens, out_dim), dtype=dtype, device=device)


class BreakablePiecewiseCudaGraphRunner(PiecewiseCudaGraphRunner):
    """Piecewise CUDA graph runner using breakable CUDA graph.

    Captures the model forward as a breakable CUDA graph with graph breaks
    at attention layers. Much simpler than the torch.compile-based approach.
    """

    def __init__(self, model_runner: ModelRunner):
        # Skip parent __init__ (which does torch.compile) — set up shared state directly
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}

        self.quant_config = getattr(model_runner.model, "quant_config", None)
        self.is_multimodal = model_runner.is_multimodal

        # Capture sizes
        capture_tokens = model_runner.server_args.piecewise_cuda_graph_tokens
        assert capture_tokens is not None
        self.capture_num_tokens = sorted(capture_tokens)
        self.max_num_tokens = (
            max(self.capture_num_tokens) if self.capture_num_tokens else 8192
        )
        self.max_bs = model_runner.req_to_token_pool.size

        log_info_on_rank0(
            logger,
            f"[Breakable PCG] Capture num tokens: {self.capture_num_tokens}",
        )

        # Reuse parent's buffer setup
        self._init_buffers(model_runner)

        self.attention_layers = model_runner.attention_layers
        self.moe_layers = model_runner.moe_layers
        self.moe_fusions = model_runner.moe_fusions

        # Bridge buffers — one set, reused across all layers and token sizes.
        attn_layer_0 = self.attention_layers[0]
        set_bridge_buffers(
            BridgeBuffers(
                self.max_num_tokens, attn_layer_0, self.device, model_runner.dtype
            )
        )

        with torch.device(self.device):
            self.static_seq_lens = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.static_extend_seq_lens = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.static_extend_prefix_lens = torch.zeros(
                (self.max_bs,), dtype=torch.int64
            )
            self.static_extend_start_loc = torch.zeros(
                (self.max_bs,), dtype=torch.int64
            )
            self.static_req_pool_indices = torch.zeros(
                (self.max_bs,), dtype=torch.int64
            )
            self.static_orig_seq_lens = torch.zeros((self.max_bs,), dtype=torch.int64)

        # Memory pool
        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        set_graph_pool_id(get_global_graph_memory_pool())

        # Warmup then capture
        self._warmup()
        self.device_module.synchronize()
        self.model_runner.tp_group.barrier()
        self._capture_all()

        self.raw_num_tokens = 0

    def _init_buffers(self, model_runner):
        """Initialize input buffers (shared logic with parent)."""
        from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
            PrefillInputBuffers,
        )
        from sglang.srt.utils import is_npu

        with torch.device(self.device):
            input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
            out_cache_loc = torch.zeros(
                (self.max_num_tokens,),
                dtype=torch.int64 if not is_npu() else torch.int32,
            )
            out_cache_loc_swa = (
                torch.zeros((self.max_num_tokens,), dtype=torch.int64)
                if model_runner.is_hybrid_swa
                else None
            )
            positions = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
            if self.is_multimodal:
                input_embeds = torch.zeros(
                    (self.max_num_tokens, model_runner.model_config.hidden_size),
                    dtype=model_runner.dtype,
                )
                mrope_positions = torch.zeros(
                    (3, self.max_num_tokens), dtype=torch.int64
                )
            else:
                input_embeds = None
                mrope_positions = None

        self.buffers = PrefillInputBuffers(
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            positions=positions,
            input_embeds=input_embeds,
            mrope_positions=mrope_positions,
        )
        self.buffers.share_buffers()

    def _run_forward(self, forward_batch, num_tokens):
        """Run model forward with proper context."""
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(None, num_tokens, forward_batch.dp_padding_mode.is_max_len())
        set_is_extend_in_batch(False)

        with set_forward_context(
            forward_batch,
            self.attention_layers,
            self.quant_config,
            self.moe_layers,
            self.moe_fusions,
        ):
            output = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
        return output

    def _build_capture_forward_batch(self, num_tokens):
        """Build a ForwardBatch for capture using static buffers for stable addresses."""
        from sglang.srt.layers.dp_attention import DpPaddingMode
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        buffers = self.buffers
        bs = 1
        self.static_seq_lens[:bs].fill_(num_tokens)
        self.static_extend_seq_lens[:bs].fill_(num_tokens)
        self.static_extend_prefix_lens[:bs].zero_()
        self.static_extend_start_loc[:bs].zero_()
        self.static_req_pool_indices[:bs].copy_(torch.arange(bs, device=self.device))
        self.static_orig_seq_lens[:bs].fill_(num_tokens)

        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=buffers.input_ids[:num_tokens],
            input_embeds=(
                buffers.input_embeds[:num_tokens] if self.is_multimodal else None
            ),
            req_pool_indices=self.static_req_pool_indices[:bs],
            seq_lens=self.static_seq_lens[:bs],
            next_token_logits_buffer=None,
            orig_seq_lens=self.static_orig_seq_lens[:bs],
            seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=buffers.out_cache_loc[:num_tokens],
            out_cache_loc_swa=(
                buffers.out_cache_loc_swa[:num_tokens]
                if buffers.out_cache_loc_swa is not None
                else None
            ),
            seq_lens_sum=num_tokens,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            encoder_lens=None,
            return_logprob=False,
            extend_num_tokens=num_tokens,
            extend_seq_lens=self.static_extend_seq_lens[:bs],
            extend_prefix_lens=self.static_extend_prefix_lens[:bs],
            extend_start_loc=self.static_extend_start_loc[:bs],
            extend_prefix_lens_cpu=torch.tensor([0], device="cpu"),
            extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
            extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
            positions=buffers.positions[:num_tokens],
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=None,
            mrope_positions=(
                buffers.mrope_positions[:, :num_tokens] if self.is_multimodal else None
            ),
            spec_algorithm=None,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=None,
            global_forward_mode=ForwardMode.EXTEND,
            lora_ids=None,
        )

    def _warmup(self):
        """Warmup the model with a forward pass."""
        num_tokens = self.capture_num_tokens[0]
        forward_batch = self._build_capture_forward_batch(num_tokens)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        self._run_forward(forward_batch, num_tokens)

    def _capture_all(self):
        """Capture breakable CUDA graphs for all token sizes."""
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            stream = graph_capture_context.stream
            pool = get_global_graph_memory_pool()

            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_num_tokens)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_num_tokens)
            )
            for num_tokens in capture_range:
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"[Breakable PCG] Capturing ({num_tokens=} {avail_mem=:.2f} GB)"
                    )

                mem_before = torch.cuda.memory_allocated(self.device) / 1024**3
                graph, output = self._capture_one(num_tokens, pool, stream)
                self.graphs[num_tokens] = graph
                self.output_buffers[num_tokens] = output
                mem_after = torch.cuda.memory_allocated(self.device) / 1024**3
                logger.info(
                    f"[Breakable PCG] num_tokens={num_tokens}: "
                    f"segments={len(graph._segments)}, "
                    f"breaks={len(graph._break_fns)}, "
                    f"mem_delta={mem_after - mem_before:.3f} GB, "
                    f"mem_total={mem_after:.3f} GB"
                )

    def _capture_one(self, num_tokens, pool, stream):
        """Capture a breakable CUDA graph for one token size."""
        forward_batch = self._build_capture_forward_batch(num_tokens)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            return self._run_forward(forward_batch, num_tokens)

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        graph = BreakableCUDAGraph()
        with BreakableCUDAGraphContext(cuda_graph=graph, pool=pool, stream=stream):
            output = run_once()

        return graph, output

    def replay(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        num_tokens = len(forward_batch.input_ids)
        index = bisect.bisect_left(self.capture_num_tokens, num_tokens)
        static_num_tokens = self.capture_num_tokens[index]

        with enable_piecewise_cuda_graph():
            # Reuse parent's buffer preparation and static forward batch construction
            static_forward_batch = self.replay_prepare(forward_batch, **kwargs)
            bs = forward_batch.batch_size

            # Update static buffers used by graph segments (esp. logits processor).
            # The graph reads from these addresses — they must have serving-time values.
            self.static_seq_lens[:bs].copy_(forward_batch.seq_lens)
            self.static_extend_seq_lens[:bs].copy_(forward_batch.extend_seq_lens)
            self.static_extend_prefix_lens[:bs].copy_(forward_batch.extend_prefix_lens)
            self.static_extend_start_loc[:bs].copy_(forward_batch.extend_start_loc)
            self.static_req_pool_indices[:bs].copy_(forward_batch.req_pool_indices)
            if forward_batch.orig_seq_lens is not None:
                self.static_orig_seq_lens[:bs].copy_(forward_batch.orig_seq_lens)

            # Set forward context and replay
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
            with set_forward_context(
                static_forward_batch,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
                num_tokens=static_num_tokens,
            ):
                self.graphs[static_num_tokens].replay()

        output = self.output_buffers[static_num_tokens]
        if isinstance(output, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                next_token_logits=output.next_token_logits[: self.raw_num_tokens],
                hidden_states=(
                    output.hidden_states[: self.raw_num_tokens]
                    if output.hidden_states is not None
                    else None
                ),
            )
        elif isinstance(output, EmbeddingPoolerOutput):
            return output
        else:
            assert isinstance(output, PPProxyTensors)
            raise NotImplementedError(
                "PPProxyTensors is not supported in BreakablePiecewiseCudaGraphRunner."
            )
