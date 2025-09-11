import bisect
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
)
from sglang.srt.models.qwen3_next import Qwen3HybridLinearDecoderLayer

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class MambaStateUpdateCudaGraphRunner:
    def __init__(self, eagle_worker: "EAGLEWorker"):
        self.eagle_worker = eagle_worker
        model_runner = eagle_worker.target_worker.model_runner
        self.model_runner = model_runner
        self.attn_backend = model_runner.attn_backend.attn_backend_list[1]
        self.req_to_token_pool = self.attn_backend.req_to_token_pool

        self.graphs = {}
        self.output_buffers = {}
        self.graph_input_buffer = None
        self.stream = torch.cuda.Stream()
        self.model = model_runner.model

        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.max_bs = self.capture_bs[-1]

        self.init_cuda_graph_state()
        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def init_cuda_graph_state(self):
        self.mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        self.num_tokens_per_bs = self.max_accepted_tokens = self.mamba_cache[2].shape[2]
        num_mamba_layers = self.mamba_cache[0].shape[0]
        conv_dtype = torch.bfloat16
        conv_shape = self.mamba_cache[0].shape[2]
        total_token_number = self.max_accepted_tokens * self.max_bs
        self.mixed_qkv_cache = torch.empty(
            size=(
                num_mamba_layers,
                total_token_number,
                conv_shape,
            ),
            dtype=conv_dtype,
            device="cuda",
        )
        self.query_start_loc = torch.zeros(
            (self.max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.state_indices = torch.zeros(
            (self.max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.has_initial_states = torch.ones(
            self.max_bs, dtype=torch.bool, device="cuda"
        )

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(self, bs: int, forward: Callable):
        """
        Capture CUDA Graph for a typical workload
        """
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        total_token_number = bs * self.max_accepted_tokens
        mixed_qkvs = self.mixed_qkv_cache[:, :total_token_number]

        query_start_loc = self.query_start_loc[: bs + 1]
        state_indices = self.state_indices[:bs]
        has_initial_states = self.has_initial_states[:bs]

        mamba_caches = self.req_to_token_pool.get_mamba_params_all_layers()
        conv_states = mamba_caches[0]
        mamba_map = self.req_to_token_pool.mamba_map

        def run_once():
            for i in range(len(self.model.model.layers)):
                layer = self.model.model.layers[i]
                if not isinstance(layer, Qwen3HybridLinearDecoderLayer):
                    continue
                conv_weights = layer.linear_attn.conv1d.weight.view(
                    layer.linear_attn.conv1d.weight.size(0),
                    layer.linear_attn.conv1d.weight.size(2),
                )
                layer_id = mamba_map[i]

                causal_conv1d_fn(
                    mixed_qkvs[layer_id].transpose(0, 1),
                    conv_weights,
                    layer.linear_attn.conv1d.bias,
                    activation=layer.linear_attn.activation,
                    conv_states=conv_states[layer_id],
                    has_initial_state=has_initial_states,
                    cache_indices=state_indices,
                    query_start_loc=query_start_loc,
                )

            return None

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

        with torch.cuda.graph(
            graph, pool=get_global_graph_memory_pool(), stream=stream
        ):
            out = run_once()

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def can_run(self, accepted_length):
        bs = accepted_length.shape[0]
        return bs <= self.max_bs

    def replay_repare(self, accepted_length):
        request_number = accepted_length.shape[0]
        # QQ: step = spec num_draft token num
        num_draft_tokens = self.req_to_token_pool.mamba_pool.mamba_cache[2].shape[2]
        query_start_loc = accepted_length.cumsum(-1, dtype=accepted_length.dtype)
        query_start_loc = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=query_start_loc.dtype,
                    device=query_start_loc.device,
                ),
                query_start_loc,
            ]
        )
        mask = torch.arange(num_draft_tokens, device=accepted_length.device).unsqueeze(
            0
        ) < accepted_length.unsqueeze(1)

        state_indices_tensor = self.attn_backend.forward_metadata.mamba_cache_indices[
            :request_number
        ]
        mamba_caches = self.req_to_token_pool.get_mamba_params_all_layers()

        _, ssm_states, mix_qkv_cache, intermediate_state_cache = mamba_caches
        mixed_qkvs = mamba_caches[2][:, state_indices_tensor][:, mask]
        self.mixed_qkv_cache[:, : mixed_qkvs.shape[1]].copy_(mixed_qkvs)
        self.query_start_loc[: request_number + 1] = query_start_loc
        self.query_start_loc[request_number + 1 :] = self.query_start_loc[
            request_number
        ]
        self.state_indices[:request_number] = state_indices_tensor
        self.state_indices[request_number:] = -1
        valid_mask = accepted_length > 0
        if intermediate_state_cache is not None:
            last_steps = (accepted_length - 1).to(torch.int64)
            valid_state_indices = state_indices_tensor[valid_mask].to(torch.int64)

            ssm_states[:, valid_state_indices, :] = intermediate_state_cache[
                :, valid_state_indices, last_steps
            ].to(ssm_states.dtype)

    def replay(self, accepted_length):
        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = accepted_length.shape[0]
        index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]

        self.replay_repare(accepted_length)
        # Replay
        self.graphs[bs].replay()
