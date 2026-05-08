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
"""Breakable CUDA graph (BCG) runner.

Captures the model forward as a sequence of ``torch.cuda.CUDAGraph`` segments
split at attention layers. Functionally parallel to the torch.compile-based
PCG runner but does not depend on torch.compile or FX graph splitting — graph
breaks are inserted eagerly via :func:`eager_on_graph` decorated callables
(radix attention for dense models, mamba for hybrid models).
"""

from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, Union

import torch
import tqdm

from sglang.srt.compilation.piecewise_context_manager import set_forward_context
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
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
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


class BreakableCudaGraphRunner:
    """Breakable CUDA graph runner.

    Captures the model forward as a series of ``torch.cuda.CUDAGraph`` segments
    with graph breaks at attention layers. Simpler than the torch.compile-based
    PCG runner: no FX tracing, no compiled-kernel fusion — just segment-level
    graph capture of the eager kernel stream.
    """

    # replay_prepare shares its buffer-population logic with the PCG runner —
    # bind the method here without inheriting. __init__, capture, and replay
    # diverge enough that inheritance would obscure more than it saves.
    replay_prepare = PiecewiseCudaGraphRunner.replay_prepare

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}

        self.quant_config = getattr(model_runner.model, "quant_config", None)
        self.is_multimodal = model_runner.is_multimodal
        # Read by the shared replay_prepare (bound from PiecewiseCudaGraphRunner).
        self.capture_return_pooled_hidden_states = not model_runner.is_generation

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
            f"[BCG] Capture num tokens: {self.capture_num_tokens}",
        )

        self._init_buffers(model_runner)

        self.attention_layers = model_runner.attention_layers
        self.moe_layers = model_runner.moe_layers
        self.moe_fusions = model_runner.moe_fusions

        # Resolve the inner transformer-stack module (the same boundary PCG draws
        # via patch_model). At replay we monkey-patch this module's forward with
        # a closure that replays the captured CUDAGraph and returns the captured
        # hidden_states; the outer model.forward then runs logits_processor /
        # pooler eagerly with the live (multi-req) forward_batch.
        language_model = getattr(
            model_runner.model, "language_model", model_runner.model
        )
        self.layer_model = (
            language_model.model
            if hasattr(language_model, "model")
            and hasattr(language_model.model, "layers")
            else language_model
        )

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
        """Initialize input buffers."""
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

    @torch.no_grad()
    def _run_forward(self, forward_batch, num_tokens):
        """Run layer-stack forward with proper context.

        Captures only the inner transformer stack (layer_model). The outer
        model.forward's tail (logits_processor / pooler) is intentionally
        excluded — it has bs-shaped kernels that would bake batch_size=1
        into the captured graph.

        ``@torch.no_grad`` mirrors the decorator on the outer ``*ForCausalLM.forward``
        (e.g. qwen3.py:507). Calling ``layer_model.forward`` directly skips that
        decorator, so we apply it here — without it some MoE @torch.compile
        kernels (``torch.sum(out=...)``) fail dynamo with "out= doesn't support
        autograd", and mamba state ops can spuriously track gradients.
        """
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
            output = self.layer_model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
        return output

    def _build_capture_forward_batch(self, num_tokens):
        """Build a bs=1 placeholder ForwardBatch for capture.

        bs=1 here is only a placeholder for attention/mamba breaks' metadata
        shapes; replay supplies live multi-req metadata via replay_prepare.
        Captured kernels run only on the token-major layer stack and are
        bs-invariant.
        """
        from sglang.srt.layers.dp_attention import DpPaddingMode
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        buffers = self.buffers
        bs = 1
        with torch.device(self.device):
            seq_lens = torch.full((bs,), num_tokens, dtype=torch.int64)
            extend_seq_lens = torch.full((bs,), num_tokens, dtype=torch.int64)
            extend_prefix_lens = torch.zeros((bs,), dtype=torch.int64)
            extend_start_loc = torch.zeros((bs,), dtype=torch.int64)
            req_pool_indices = torch.arange(bs, dtype=torch.int64)
            orig_seq_lens = torch.full((bs,), num_tokens, dtype=torch.int64)

        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=buffers.input_ids[:num_tokens],
            input_embeds=(
                buffers.input_embeds[:num_tokens] if self.is_multimodal else None
            ),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            next_token_logits_buffer=None,
            orig_seq_lens=orig_seq_lens,
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
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
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
        ), graph_capture() as graph_capture_context, enable_breakable_cuda_graph():
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
                        f"[BCG] Capturing ({num_tokens=} {avail_mem=:.2f} GB)"
                    )

                graph, output = self._capture_one(num_tokens, pool, stream)
                self.graphs[num_tokens] = graph
                self.output_buffers[num_tokens] = output

    def can_run(self, forward_batch: "ForwardBatch"):
        if forward_batch.forward_mode.is_target_verify():
            return False
        if forward_batch.input_embeds is not None:
            return False
        if forward_batch.replace_embeds is not None:
            return False
        num_tokens = len(forward_batch.input_ids)
        if forward_batch.return_logprob:
            for start_len, seq_len in zip(
                forward_batch.extend_logprob_start_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
            ):
                if start_len is not None and start_len < seq_len:
                    return False
        return num_tokens <= self.max_num_tokens

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
        with BreakableCUDAGraphCapture(cuda_graph=graph, pool=pool, stream=stream):
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

        captured_graph = self.graphs[static_num_tokens]
        captured_hidden = self.output_buffers[static_num_tokens]

        # Closure replaces layer_model.forward for the duration of the outer
        # model.forward call. Replays the captured CUDAGraph and hands the
        # outer forward the captured hidden_states; logits_processor / pooler
        # then runs eagerly on top with the live multi-req forward_batch.
        def replay_layer_forward(*args, **layer_kwargs):
            captured_graph.replay()
            return captured_hidden

        with enable_breakable_cuda_graph():
            static_forward_batch = self.replay_prepare(forward_batch, **kwargs)

            original_layer_forward = self.layer_model.forward
            self.layer_model.forward = replay_layer_forward
            try:
                self.model_runner.attn_backend.init_forward_metadata(forward_batch)
                with set_forward_context(
                    static_forward_batch,
                    self.attention_layers,
                    self.quant_config,
                    self.moe_layers,
                    self.moe_fusions,
                ):
                    output = self.model_runner.model.forward(
                        static_forward_batch.input_ids,
                        static_forward_batch.positions,
                        static_forward_batch,
                        **kwargs,
                    )
            finally:
                self.layer_model.forward = original_layer_forward

        if isinstance(output, LogitsProcessorOutput):
            # Slice trailing-padding off hidden_states; next_token_logits is
            # bs-shaped from logits_processor (bs <= raw_num_tokens), so the
            # slice is a no-op for that field but matches PCG's pattern.
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
                "PPProxyTensors is not supported in BreakableCudaGraphRunner."
            )
