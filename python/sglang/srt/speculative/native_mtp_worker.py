from __future__ import annotations

import copy
from dataclasses import replace

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    resolve_attention_backend_strs,
)
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    resolve_layer_indices,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import empty_context


class NativeMTPModelRunner(ModelRunner):
    """Small ModelRunner-compatible facade over target_model.mtp."""

    def __init__(self, target_runner: ModelRunner, server_args: ServerArgs) -> None:
        self._target_runner = target_runner
        self.server_args = server_args
        self.model = target_runner.model.mtp
        self.model_config = copy.copy(target_runner.model_config)
        mtp_hf_config = copy.deepcopy(self.model.model.config)
        mtp_hf_config.architectures = ["Qwen3NextForCausalLMMTP"]
        mtp_hf_config.num_nextn_predict_layers = 1
        self.model_config.hf_config = mtp_hf_config
        self.model_config.hf_text_config = mtp_hf_config.get_text_config()
        self.model_config.num_hidden_layers = 1
        self.model_config.num_attention_layers = 1
        self.model_config.num_nextn_predict_layers = 1
        setattr(self.model, "num_stages", 1)
        self.device = target_runner.device
        self.gpu_id = target_runner.gpu_id
        self.ps = replace(target_runner.ps, pp_rank=0)
        self.pp_group = target_runner.pp_group
        self.tp_group = target_runner.tp_group
        self.dtype = target_runner.dtype
        self.is_draft_worker = True
        self.draft_attention_backend = server_args.speculative_draft_attention_backend
        resolved_backend_strs = resolve_attention_backend_strs(model_runner=self)
        self.prefill_attention_backend_str = resolved_backend_strs.prefill
        self.decode_attention_backend_str = resolved_backend_strs.decode
        self.attn_backend = None
        self.decode_attn_backend = None
        self.decode_attn_backend_group = []
        self.draft_attn_backend = None
        self.use_mla_backend = target_runner.use_mla_backend
        self.attention_chunk_size = target_runner.attention_chunk_size
        self.page_size = target_runner.page_size
        self.sliding_window_size = target_runner.sliding_window_size
        self.is_hybrid_swa = target_runner.is_hybrid_swa
        self.is_hybrid_swa_compress = target_runner.is_hybrid_swa_compress
        self.kv_cache_dtype = target_runner.kv_cache_dtype
        self.kv_cache_dtype_str = target_runner.kv_cache_dtype_str
        self.pre_model_load_memory = target_runner.pre_model_load_memory
        self.spec_aux_config = target_runner.spec_aux_config
        self.forward_stream = target_runner.forward_stream
        self.draft_model_idx = 0
        self.enable_hisparse = target_runner.enable_hisparse
        self._token_oracle_manager = None
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.layer_info = resolve_layer_indices(
            model=self.model,
            model_config=self.model_config,
            is_draft_worker=True,
            spec_algorithm=self.spec_algorithm,
        )
        self.ngram_embedding_manager = None
        self.lora_manager = None
        self.canary_manager = None
        self.hisparse_coordinator = None
        self.decode_cuda_graph_runner = None
        self.prefill_cuda_graph_runner = None
        self.graph_memory_usage: dict[str, float] = {}
        self.graph_time_usage: dict[str, float] = {}
        self.weight_load_time = 0.0
        self.forward_pass_id = 0
        self.max_running_requests = server_args.max_running_requests
        self.mtp_draft_device_pools = ()
        self.init_new_workspace = False
        self.enable_elastic_ep = False
        self.eplb_manager = None

        self.memory_pool_config = target_runner.memory_pool_config
        self.req_to_token_pool = target_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = target_runner.token_to_kv_pool_allocator
        self.token_to_kv_pool = None

    def __getattr__(self, name: str):
        return getattr(self._target_runner, name)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        self.memory_pool_config = memory_pool_config
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        ModelRunner.alloc_memory_pool(self, memory_pool_config)

    def forward(self, forward_batch: ForwardBatch) -> ModelRunnerOutput:
        self.forward_pass_id += 1
        attn_backend = self.attn_backend or self.draft_attn_backend
        if attn_backend is None:
            raise RuntimeError("Native MTP attention backend is not initialized.")
        with forward_context(ForwardContext(attn_backend=attn_backend)):
            ModelRunner._prepare_eager_forward_batch(self, forward_batch)
            if forward_batch.needs_forward_metadata_init():
                if hasattr(self.model, "prepare_forward_batch"):
                    self.model.prepare_forward_batch(forward_batch)
                attn_backend.init_forward_metadata(forward_batch)
            ret = self.model(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                input_embeds=forward_batch.input_embeds,
            )
            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

        return ModelRunnerOutput(logits_output=ret, can_run_graph=False)


class NativeMTPDraftWorker(EagleDraftWorker):
    """EAGLE draft worker that runs an embedded target-model MTP block."""

    def _build_draft_worker(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        del server_args, gpu_id, ps, nccl_port, target_worker
        return None

    def _build_draft_runner(self):
        target_runner = self.target_worker.model_runner
        if getattr(target_runner.model, "mtp", None) is None:
            raise ValueError(
                "--enable-native-mtp requires the target model to initialize "
                "an embedded mtp block."
            )

        return NativeMTPModelRunner(target_runner, self.server_args)

    def _build_draft_tp_context(self):
        return empty_context

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_runner.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_token_map()
        self.init_lm_head()

    def init_attention_backends(self):
        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()

    def init_attention_backend(self):
        super().init_attention_backend()
        if self.draft_runner.attn_backend is None:
            self.draft_runner.attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()


class NativeMTPWorkerV2(EAGLEWorkerV2):
    """EAGLE v2 orchestration with native embedded MTP as the draft side."""

    def _build_draft_worker(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        return NativeMTPDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )
