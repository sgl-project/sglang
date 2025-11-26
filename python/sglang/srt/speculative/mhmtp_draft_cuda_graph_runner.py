"""MHMTP Draft Model CUDA Graph Runner for Speculative Decoding.

This module provides efficient CUDA graph capture and replay for multi-step
draft token generation using the MHMTP speculative decoding algorithm.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.mhmtp_utils import MhmtpDraftInput, fast_topk
from sglang.srt.speculative.spec_utils import select_top_k_tokens
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.mhmtp_worker import MhmtpWorker


class MultiLayerForwardOutput:
    """Container for outputs from N forward passes in multi-step generation.

    Stores tree information (scores, tokens, parents) from N consecutive
    forward passes through the draft model.

    Attributes:
        num_layers: Number of forward passes/layers.
        tree_infos: List of (scores, tokens, parents) tuples for each layer.
    """

    def __init__(self, num_layers: int):
        """Initialize MultiLayerForwardOutput with variable number of layers.

        Args:
            num_layers: Number of forward passes/layers.
        """
        self.num_layers = num_layers
        # Store tree info as list: tree_infos[i] = (scores, tokens, parents)
        self.tree_infos = [None] * num_layers

    def set_layer(self, layer_idx: int, scores, tokens, parents):
        """Set tree info for a layer.

        Args:
            layer_idx: The layer index.
            scores: Token scores for this layer.
            tokens: Token indices for this layer.
            parents: Parent relationships for this layer.
        """
        self.tree_infos[layer_idx] = (scores, tokens, parents)

    def __getitem__(self, idx: int):
        """Support indexing to access layers.

        Args:
            idx: The layer index.

        Returns:
            Tuple of (scores, tokens, parents) for the layer.
        """
        return self.tree_infos[idx]


class MHMTPDraftCudaGraphRunner:
    """CUDA Graph runner for MHMTP draft model multi-step generation.

    Captures and replays CUDA graphs for efficient execution of N consecutive
    forward passes through the draft model, enabling fast speculative token generation.
    Supports arbitrary number of forward passes through configurable parameters.
    Requires speculative_num_steps >= 3.
    """

    def __init__(self, mhmtp_worker: MhmtpWorker, mtp_layer: int):
        """Initialize the CUDA graph runner.

        Args:
            mhmtp_worker: The mhmtp worker instance containing draft model.
            mtp_layer: Multi-token prediction layer index.

        Raises:
            ValueError: If speculative_num_steps < 3.
            Exception: If CUDA graph capture fails.
        """
        self.mhmtp_worker = mhmtp_worker
        self.model_runner = mhmtp_worker.model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.mtp_layer = mtp_layer

        # Configuration from server args
        self.enable_torch_compile = self.model_runner.server_args.enable_torch_compile
        self.disable_padding = self.model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(
            self.model_runner.server_args
        )
        self.require_mlp_tp_gather = require_mlp_tp_gather(
            self.model_runner.server_args
        )
        self.require_mlp_sync = require_mlp_sync(self.model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(
            self.model_runner.server_args
        )

        # Parallelism configuration
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.server_args.dp_size

        # Speculative decoding parameters
        self.speculative_num_steps = self.model_runner.server_args.speculative_num_steps

        # Validate speculative_num_steps >= 3
        if self.speculative_num_steps < 3:
            raise ValueError(
                f"speculative_num_steps must be >= 3, got {self.speculative_num_steps}"
            )

        self.topk = self.model_runner.server_args.speculative_eagle_topk
        self.num_tokens_per_bs = self.speculative_num_steps + 1

        # Use variable instead of hardcoded 3
        self.num_forward_passes = self.speculative_num_steps

        # Graph capture configuration
        self.enable_profile_cuda_graph = (
            self.model_runner.server_args.enable_profile_cuda_graph
        )
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(self.model_runner)
        self.padded_static_len = -1

        # Maximum sizes for tensor allocation
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        # Initialize attention backends for all N layers
        self._init_attention_backends()

        # Allocate input and output tensors
        self._allocate_graph_tensors()

        # Capture CUDA graphs
        if self.enable_torch_compile:
            set_torch_compile_config()

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _init_attention_backends(self) -> None:
        """Initialize CUDA graph state for all N attention backends."""
        attn_backends = (
            self.mhmtp_worker.draft_model_runner.draft_attn_backend.attn_backends
        )

        for i in range(self.num_forward_passes):
            backend = attn_backends[i]
            backend.init_cuda_graph_state(self.max_bs, self.max_num_token)
            if i == 0:
                self.seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()

    def _allocate_graph_tensors(self) -> None:
        """Allocate GPU tensors for CUDA graph capture and replay."""
        with torch.device("cuda"):
            # Input and hidden tensors for N layers
            self.input_ids_layers = [
                torch.zeros((self.max_num_token,), dtype=torch.int64)
                for _ in range(self.num_forward_passes)
            ]

            hidden_size = self.model_runner.model_config.hidden_size
            dtype = self.model_runner.dtype

            self.hidden_states_layers = [
                torch.zeros((self.max_num_token, hidden_size), dtype=dtype)
                for _ in range(self.num_forward_passes)
            ]

            # Batch-level tensors (shared across all layers)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.ones((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)

            # Sequence length tensors
            self.seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            self.extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            self.accept_length = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            self.seq_lens_cpu = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )

            # Gather buffers for distributed execution
            if self.require_gathered_buffer:
                self.gathered_buffer = torch.zeros(
                    (self.max_num_token, hidden_size),
                    dtype=dtype,
                )
                if self.require_mlp_tp_gather:
                    self.global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    self.global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """Check if the forward batch can be executed with this runner.

        Args:
            forward_batch: The forward batch to check.

        Returns:
            True/False for mhmtp CUDA graph runner.
        """
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                sum(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
            )
        else:
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def capture(self) -> None:
        """Capture CUDA graphs for all batch sizes."""
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, bs: int, forward: Callable
    ) -> tuple[torch.cuda.CUDAGraph, MultiLayerForwardOutput]:
        """Capture CUDA graph for one batch size.

        Performs N forward passes through the draft model, capturing
        the computation graph. Uses streaming token selection to build
        a tree of candidate tokens.

        Args:
            bs: Batch size to capture.
            forward: Forward function (unused, for compatibility).

        Returns:
            Tuple of (CUDA graph, MultiLayerForwardOutput with tree information).
        """
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Prepare graph inputs
        input_ids = self.input_ids_layers[0][:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        extend_seq_lens = self.extend_seq_lens[:bs]
        accept_length = self.accept_length[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        hidden_states = self.hidden_states_layers[0][:num_tokens]

        # Setup distributed synchronization tensors
        global_num_tokens, gathered_buffer, global_num_tokens_for_logprob = (
            self._setup_distributed_tensors(num_tokens)
        )

        # Create forward batch for graph capture
        forward_batch = self._create_forward_batch_for_capture(
            bs=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            positions=positions,
            extend_seq_lens=extend_seq_lens,
            hidden_states=hidden_states,
            accept_length=accept_length,
            num_tokens=num_tokens,
            global_num_tokens=global_num_tokens,
            global_num_tokens_for_logprob=global_num_tokens_for_logprob,
            gathered_buffer=gathered_buffer,
        )

        # Define computation to capture
        def run_once() -> MultiLayerForwardOutput:
            """Execute N forward passes and return tree information."""
            return self._run_multi_forward_passes(forward_batch, bs, num_tokens)

        # Warmup runs
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        # Capture graph
        with torch.cuda.graph(
            graph, pool=get_global_graph_memory_pool(), stream=stream
        ):
            out = run_once()

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def _setup_distributed_tensors(self, num_tokens: int) -> tuple:
        """Setup tensors for distributed synchronization.

        Args:
            num_tokens: Total number of tokens in the batch.

        Returns:
            Tuple of (global_num_tokens, gathered_buffer, global_num_tokens_for_logprob).
        """
        if not self.require_gathered_buffer:
            return None, None, None

        gathered_buffer = self.gathered_buffer[:num_tokens]

        if self.require_mlp_tp_gather:
            token_per_dp = [
                num_tokens // self.dp_size + (i < (num_tokens % self.dp_size))
                for i in range(self.dp_size)
            ]
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    token_per_dp,
                    dtype=torch.int32,
                    device=self.input_ids_layers[0].device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    token_per_dp,
                    dtype=torch.int32,
                    device=self.input_ids_layers[0].device,
                )
            )
        else:
            assert self.require_attn_tp_gather
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.input_ids_layers[0].device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.input_ids_layers[0].device,
                )
            )

        return (
            self.global_num_tokens_gpu,
            gathered_buffer,
            self.global_num_tokens_for_logprob_gpu,
        )

    def _create_forward_batch_for_capture(
        self,
        bs: int,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        positions: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        hidden_states: torch.Tensor,
        accept_length: torch.Tensor,
        num_tokens: int,
        global_num_tokens: torch.Tensor,
        global_num_tokens_for_logprob: torch.Tensor,
        gathered_buffer: torch.Tensor,
    ) -> ForwardBatch:
        """Create forward batch for CUDA graph capture.

        Args:
            bs: Batch size.
            input_ids: Input token IDs.
            req_pool_indices: Request pool indices.
            seq_lens: Sequence lengths.
            out_cache_loc: Output cache locations.
            positions: Position indices.
            extend_seq_lens: Extended sequence lengths.
            hidden_states: Hidden states from previous layer.
            accept_length: Accepted token lengths.
            num_tokens: Total number of tokens.
            global_num_tokens: Global token counts for distributed execution.
            global_num_tokens_for_logprob: Global token counts for logprob.
            gathered_buffer: Gathered buffer for distributed execution.

        Returns:
            ForwardBatch configured for graph capture.
        """
        spec_info = MhmtpDraftInput(
            hidden_states=hidden_states,
            accept_length=accept_length,
        )
        spec_info.positions = None

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            attn_backend=(
                self.mhmtp_worker.draft_model_runner.draft_attn_backend.attn_backends[0]
            ),
            extend_seq_lens=extend_seq_lens,
            padded_static_len=self.padded_static_len,
        )

        # Initialize forward metadata for all N layers
        attn_backends = (
            self.mhmtp_worker.draft_model_runner.draft_attn_backend.attn_backends
        )
        for i in range(self.num_forward_passes):
            attn_backends[i].init_forward_metadata_capture_cuda_graph(
                bs=bs,
                num_tokens=num_tokens,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DRAFT_EXTEND,
                spec_info=spec_info,
            )

        forward_batch.mtp_index = 0
        return forward_batch

    def _run_multi_forward_passes(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        num_tokens: int,
    ) -> MultiLayerForwardOutput:
        """Execute N sequential forward passes through draft model.

        N must be >= 3 (validated in __init__).
        """
        scores = None
        attn_backends = (
            self.mhmtp_worker.draft_model_runner.draft_attn_backend.attn_backends
        )

        output = MultiLayerForwardOutput(self.num_forward_passes)
        topk_indices_per_layer = []

        # Run N forward passes
        for layer_idx in range(self.num_forward_passes):
            forward_batch.mtp_index = layer_idx
            forward_batch.attn_backend = attn_backends[layer_idx]

            # Forward pass
            ret = self.mhmtp_worker.draft_model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

            # Get probabilities and top-k tokens
            probs = torch.softmax(ret.next_token_logits[3::4], dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_indices_per_layer.append(topk_index)

            # Update scores for tree
            _, _, scores, tree_info = select_top_k_tokens(
                layer_idx,
                topk_p,
                topk_index,
                ret.hidden_states[3::4],
                scores,
                self.topk,
            )

            # Store tree info for this layer
            output.set_layer(layer_idx, tree_info[0], tree_info[1], tree_info[2])

            # Prepare input for next layer (if not last layer)
            if layer_idx < self.num_forward_passes - 1:
                next_layer_idx = layer_idx + 1

                input_view = forward_batch.input_ids.view(bs, self.num_tokens_per_bs)
                next_input_view = self.input_ids_layers[next_layer_idx][
                    :num_tokens
                ].view(bs, self.num_tokens_per_bs)

                # For layer i -> layer i+1:
                # - Keep the last (num_tokens_per_bs - i - 1) original tokens
                # - Add topk tokens from layers 0..i in reverse order (i down to 0)

                num_keep = self.num_tokens_per_bs - layer_idx - 1

                # Copy last num_keep tokens from current input to next input
                if num_keep > 0:
                    next_input_view[:, :num_keep].copy_(input_view[:, -(num_keep):])

                # Add topk tokens at the end in reverse order
                for j in range(layer_idx + 1):
                    # Position: num_keep + (layer_idx - j)
                    # Token: topk_indices_per_layer[layer_idx - j]
                    next_input_view[:, num_keep + layer_idx - j].copy_(
                        topk_indices_per_layer[layer_idx - j].view(bs, 1)[:, 0]
                    )

                self.hidden_states_layers[next_layer_idx][:num_tokens].copy_(
                    ret.hidden_states
                )

                # Update for next iteration
                self.input_ids_layers[0].copy_(self.input_ids_layers[next_layer_idx])
                self.hidden_states_layers[0].copy_(
                    self.hidden_states_layers[next_layer_idx]
                )

        return output

    def replay(self, forward_batch: ForwardBatch) -> MultiLayerForwardOutput:
        """Replay captured CUDA graph for forward batch execution.

        Args:
            forward_batch: The forward batch to execute.

        Returns:
            MultiLayerForwardOutput with tree information from N forward passes.
        """
        raw_bs = forward_batch.batch_size
        num_tokens = forward_batch.input_ids.shape[0]

        # Find appropriate batch size for graph execution
        if self.require_mlp_tp_gather:
            total_batch_size = (
                sum(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
            )
            index = bisect.bisect_left(self.capture_bs, total_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]

        # Reset padding tensors if batch size changed
        if bs * self.num_tokens_per_bs != num_tokens:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()
            self.accept_length.fill_(1)
            self.extend_seq_lens.fill_(1)

        # Copy batch inputs to graph tensors
        self._copy_batch_inputs_to_graph(forward_batch, raw_bs, bs, num_tokens)

        # Initialize attention metadata for all N layers
        self._init_attention_metadata_for_replay(forward_batch, raw_bs, bs, num_tokens)

        # Replay CUDA graph
        self.graphs[bs].replay()

        # Extract and trim outputs to actual batch size
        out = self.output_buffers[bs]
        if bs != raw_bs:
            forward_batch.spec_info.accept_length = self.accept_length[:raw_bs]

        # Trim all layers to actual batch size
        trimmed_output = MultiLayerForwardOutput(self.num_forward_passes)
        for i in range(self.num_forward_passes):
            scores, tokens, parents = out[i]
            trimmed_output.set_layer(
                i, scores[:raw_bs], tokens[:raw_bs], parents[:raw_bs]
            )

        return trimmed_output

    def _copy_batch_inputs_to_graph(
        self,
        forward_batch: ForwardBatch,
        raw_bs: int,
        bs: int,
        num_tokens: int,
    ) -> None:
        """Copy batch inputs to graph tensors.

        Args:
            forward_batch: The forward batch to copy from.
            raw_bs: Raw batch size.
            bs: Padded batch size for graph execution.
            num_tokens: Total number of tokens.
        """
        self.input_ids_layers[0][:num_tokens].copy_(forward_batch.input_ids)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)

        if forward_batch.extend_seq_lens is not None:
            self.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)

        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        self.positions[:num_tokens].copy_(forward_batch.positions)
        self.hidden_states_layers[0][:num_tokens].copy_(
            forward_batch.spec_info.hidden_states
        )

        if forward_batch.spec_info.accept_length is not None:
            self.accept_length[:raw_bs].copy_(forward_batch.spec_info.accept_length)

        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)

        # Copy distributed execution tensors
        if self.require_gathered_buffer:
            self.global_num_tokens_gpu.copy_(forward_batch.global_num_tokens_gpu)
            self.global_num_tokens_for_logprob_gpu.copy_(
                forward_batch.global_num_tokens_for_logprob_gpu
            )
            forward_batch.gathered_buffer = self.gathered_buffer

        # Copy CPU sequence lengths
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        # Update spec info with padded tensors
        if bs != raw_bs:
            forward_batch.spec_info.positions = self.positions[:num_tokens]
            forward_batch.spec_info.accept_length = self.accept_length[:bs]

    def _init_attention_metadata_for_replay(
        self,
        forward_batch: ForwardBatch,
        raw_bs: int,
        bs: int,
        num_tokens: int,
    ) -> None:
        """Initialize attention metadata for all N layers before replay.

        Args:
            forward_batch: The forward batch being replayed.
            raw_bs: Raw batch size.
            bs: Padded batch size.
            num_tokens: Total number of tokens.
        """
        attn_backends = (
            self.mhmtp_worker.draft_model_runner.draft_attn_backend.attn_backends
        )

        padded_seq_lens_sum = (
            forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value
        )

        for i in range(self.num_forward_passes):
            attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs=bs,
                req_pool_indices=self.req_pool_indices,
                seq_lens=self.seq_lens,
                seq_lens_sum=padded_seq_lens_sum,
                encoder_lens=None,
                forward_mode=ForwardMode.DRAFT_EXTEND,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=self.seq_lens_cpu,
            )
