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
"""ModelRunner runs the forward passes of the models."""

from __future__ import annotations

import contextlib
import datetime
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    mambaish_config,
)
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
)
from sglang.srt.distributed.bootstrap import init_torch_distributed
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    maybe_init_shared_mooncake_transfer_engine,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    join_process_groups,
    try_recover_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.cp.utils import (
    get_cp_strategy,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.model_parallel import apply_torch_tp
from sglang.srt.layers.n_gram_embedding_manager import NgramEmbeddingManager
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import (
    LoRAManager,
    init_lora_cuda_graph_moe_buffers,
)
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
)
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    build_attention_backends,
    configure_aux_hidden_state_capture,
    get_attention_backend,
)
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_cuda_graphs,
    capture_decode_graph,
    capture_prefill_graph,
)
from sglang.srt.model_executor.model_runner_components.expert_location_helpers import (
    get_healthy_expert_location_src_rank,
)
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    ModelLayerInfo,
    adjust_hybrid_swa_layer_ids,
    resolve_layer_indices,
)
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_kv_cache_scales,
    maybe_downgrade_dtype_for_legacy_gpu,
    maybe_register_debug_tensor_dump_hook,
    maybe_trigger_remote_instance_nccl_send_group,
    report_online_quantization,
    resolve_sliding_window_size,
)
from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
    init_lplb_solvers,
    prepare_moe_topk,
)
from sglang.srt.model_executor.model_runner_components.msprobe import (
    create_msprobe_debugger,
)
from sglang.srt.model_executor.model_runner_components.pool_configurator import (
    MemoryPoolConfig,
)
from sglang.srt.model_executor.model_runner_components.pp_proxy import (
    resolve_pp_proxy_topk_size,
)
from sglang.srt.model_executor.model_runner_components.quantization_checks import (
    check_quantized_moe_compatibility,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transport import (
    RemoteInstanceWeightTransport,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    resolve_spec_aux_hidden_state_config,
)
from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_executor.runner import (
    EagerRunner,
)
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.platforms import current_platform
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    disable_routed_experts_capture_for_draft,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    broadcast_pyobj,
    cpu_has_amx_support,
    enable_show_time_cost,
    get_available_gpu_memory,
    is_host_cpu_arm64,
    is_npu,
    require_gathered_buffer,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.numa_utils import init_threads_binding
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.profile_utils import build_step_span_name
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker

_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)


@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.dcp_size = server_args.dcp_size
        self.dcp_rank = tp_rank % self.dcp_size
        dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                tp_rank,
                tp_size,
                dp_size,
                server_args.attn_cp_size,
            )
        )
        self.ps = ParallelState(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=server_args.attn_cp_size,
            attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            moe_dp_rank=moe_dp_rank,
            moe_dp_size=server_args.moe_dp_size,
            dcp_size=server_args.dcp_size,
            gpu_id=self.gpu_id,
        )
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = getattr(
            model_config, "is_hybrid_swa_compress", False
        )
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.init_remote_instance_weight_transport()

        self.msprobe_debugger = create_msprobe_debugger(server_args)

        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.init_spec_aux_hidden_state()

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Set the global server_args in the scheduler process
        set_global_server_args_for_scheduler(server_args)

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.local_omp_cpuid = init_threads_binding(
                tp_rank=self.ps.tp_rank, tp_size=self.ps.tp_size
            )

        # Set float32 matmul precision
        if server_args.enable_tf32_matmul:
            torch.set_float32_matmul_precision("high")

        # Get available memory before model loading.
        # Stored for later use by alloc_memory_pool().
        result = init_torch_distributed(
            server_args=self.server_args,
            model_config=self.model_config,
            device=self.device,
            ps=self.ps,
            dist_port=self.dist_port,
            is_draft_worker=self.is_draft_worker,
            local_omp_cpuid=self.local_omp_cpuid if self.device == "cpu" else None,
        )
        self.tp_group = result.tp_group
        self.pp_group = result.pp_group
        self.attention_tp_group = result.attention_tp_group
        self.pre_model_load_memory = result.pre_model_load_memory

        # Initialize MooncakeTransferEngine
        maybe_init_shared_mooncake_transfer_engine(
            server_args=self.server_args, gpu_id=self.gpu_id
        )

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # WAR fast-path: a decode-graph forward publishes a fresh event here after
        # load_batch; the scheduler's WAR barrier waits on it (then clears it)
        # instead of the whole-forward wait_stream. None -> whole-forward fallback.
        self.war_fastpath_read_done_event: Optional[torch.cuda.Event] = None

        # CPU offload
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        self.hisparse_coordinator = None

        # Load model weights and configure
        self.initialize()
        check_quantized_moe_compatibility(
            model_config=self.model_config,
            tp_size=self.ps.tp_size,
            moe_ep_size=self.ps.moe_ep_size,
            moe_dp_size=self.ps.moe_dp_size,
        )

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            )
            ElasticEPStateManager.instance().reset()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.ps.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self.init_weight_updater()
        self.init_weight_exporter()

    def init_weight_updater(self):
        self.weight_updater = WeightUpdater(
            tp_rank=self.ps.tp_rank,
            device=self.device,
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            custom_weight_loaders=self.server_args.custom_weight_loader,
            get_model=lambda: self.model,
            update_model_fields=self.update_model_fields,
            recapture_cuda_graph=self.init_decode_cuda_graph,
            get_model_runner=lambda: self,
        )

    def init_spec_aux_hidden_state(self):
        self.spec_aux_config: SpecAuxHiddenStateConfig = (
            resolve_spec_aux_hidden_state_config(
                server_args=self.server_args,
                model_config=self.model_config,
                spec_algorithm=self.spec_algorithm,
                is_draft_worker=self.is_draft_worker,
            )
        )

    def init_weight_exporter(self):
        self.weight_exporter = WeightExporter(_model_runner=self)

    def init_remote_instance_weight_transport(self):
        self.remote_instance_weight_transport = RemoteInstanceWeightTransport(
            server_args=self.server_args,
            get_model=lambda: self.model,
            tp_rank=self.ps.tp_rank,
            gpu_id=self.gpu_id,
        )

    def init_ngram_embedding_manager(self):
        self.ngram_embedding_manager = NgramEmbeddingManager.from_model(
            model=self.model,
            model_config=self.model_config,
            req_to_token_pool=self.req_to_token_pool,
            server_args=self.server_args,
            max_running_requests=self.max_running_requests,
            device=self.device,
        )

    def init_kv_cache_configurator(self):
        self.kv_cache_configurator = KVCacheConfigurator(
            device=self.device,
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            server_args=self.server_args,
            kv_cache_dtype=self.kv_cache_dtype,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=self.is_draft_worker,
            dflash_draft_num_layers=self.spec_aux_config.dflash_draft_num_layers,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_swa_compress=self.is_hybrid_swa_compress,
            use_mla_backend=self.use_mla_backend,
            mambaish_config=mambaish_config(self.model_config),
            hybrid_gdn_config=hybrid_gdn_config(self.model_config),
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            num_effective_layers=self.layer_info.num_effective_layers,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.ps.tp_size * self.ps.pp_size,
                rank=self.ps.tp_size * self.ps.pp_rank + self.ps.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self):
        server_args = self.server_args

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_weight_transport.init_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.ps.moe_ep_rank,
                )
            )
            if self.ps.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.ps.tp_rank,
                )
            )

        if self.server_args.ep_dispatch_algorithm == "lp" and not self.is_draft_worker:
            init_lplb_solvers(model_config=self.model_config)

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        if self.server_args.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)
        self._token_oracle_manager = install_token_oracle_from_env(
            server_args=server_args,
            vocab_size=self.model_config.vocab_size,
        )
        # Load the model
        self.sampler = create_sampler()
        self.load_model()
        prepare_moe_topk(
            model=self.model,
            model_config=self.model_config,
            server_args=self.server_args,
            moe_ep_size=self.ps.moe_ep_size,
            moe_ep_rank=self.ps.moe_ep_rank,
        )

        # Must run before backend/graph init so no draft graph records a
        # routed-experts capture-write kernel.
        if self.is_draft_worker:
            disable_routed_experts_capture_for_draft(self.model)

        # Load the expert backup client
        self.expert_backup_client = (
            ExpertBackupClient(self.server_args, self)
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )

        self.remote_instance_weight_transport.maybe_register_and_publish_weight_info()

        self.layer_info: ModelLayerInfo = resolve_layer_indices(
            model=self.model,
            model_config=self.model_config,
            is_draft_worker=self.is_draft_worker,
            spec_algorithm=self.spec_algorithm,
        )

        adjust_hybrid_swa_layer_ids(
            model_config=self.model_config,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            is_hybrid_swa=self.is_hybrid_swa,
        )

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.ps.tp_size > 1 and supports_torch_tp:
            apply_torch_tp(
                model=self.model, device=self.device, tp_size=self.ps.tp_size
            )

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        self.server_args.kv_cache_dtype, self.kv_cache_dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype=self.server_args.kv_cache_dtype,
            model=self.model,
            model_dtype=self.dtype,
        )

    def get_pp_proxy_topk_size(self) -> Optional[int]:
        return resolve_pp_proxy_topk_size(
            model_config=self.model_config,
            pp_size=self.ps.pp_size,
            pp_rank=self.ps.pp_rank,
            start_layer=self.layer_info.start_layer,
        )

    def alloc_memory_pool(self, memory_pool_config: Optional[MemoryPoolConfig] = None):
        """Allocate KV cache memory pools only (no backends or cuda graphs)."""
        if memory_pool_config is not None:
            self.memory_pool_config = memory_pool_config

        self.init_kv_cache_configurator()
        result = self.kv_cache_configurator.configure(
            pre_model_load_memory=self.pre_model_load_memory
        )
        self.max_total_num_tokens = result.max_total_num_tokens
        self.max_running_requests = result.max_running_requests
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool = result.token_to_kv_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.memory_pool_config = result.memory_pool_config
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = result.full_max_total_num_tokens
            self.swa_max_total_num_tokens = result.swa_max_total_num_tokens

        # Must be called AFTER init_memory_pool so the pool object exists for
        # canary to monkey-patch, and BEFORE init_decode_cuda_graph so warmup
        # forwards captured into the graph see the patched pool methods.
        self.canary_manager = install_canary(
            server_args=self.server_args,
            model_runner=self,
            token_oracle_manager=self._token_oracle_manager,
        )

        # Init ngram embedding token table
        self.init_ngram_embedding_manager()

        if self.enable_hisparse:
            from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            hisparse_cfg = parse_hisparse_config(self.server_args)
            hisparse_top_k = getattr(
                self.model_config.hf_text_config, "index_topk", hisparse_cfg.top_k
            )
            self.hisparse_coordinator = HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                top_k=hisparse_top_k,
                device_buffer_size=hisparse_cfg.device_buffer_size,
                device=self.device,
                tp_group=(
                    self.attention_tp_group.cpu_group
                    if self.server_args.enable_dp_attention
                    else self.tp_group.cpu_group
                ),
                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
            )

        self.init_routed_experts_capturer()
        self.init_indexer_capturer()

    def init_attention_backends(self):
        """Initialize attention backends only (no cuda graph capture)."""
        # Must be called BEFORE init_decode_cuda_graph() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        configure_aux_hidden_state_capture(
            model=self.model,
            eagle_use_aux_hidden_state=self.spec_aux_config.eagle_use_aux_hidden_state,
            eagle_aux_hidden_state_layer_ids=self.spec_aux_config.eagle_aux_hidden_state_layer_ids,
            dflash_use_aux_hidden_state=self.spec_aux_config.dflash_use_aux_hidden_state,
            dflash_target_layer_ids=self.spec_aux_config.dflash_target_layer_ids,
        )
        backends = build_attention_backends(model_runner=self)
        self.attn_backend = backends.attn_backend
        self.decode_attn_backend = backends.decode_attn_backend
        self.decode_attn_backend_group = backends.decode_attn_backend_group
        self.prefill_attention_backend_str = backends.prefill_attention_backend_str
        self.decode_attention_backend_str = backends.decode_attention_backend_str

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        capture = capture_cuda_graphs(
            model_runner=self, capture_decode_cuda_graph=capture_decode_cuda_graph
        )
        self.eager_runner = capture.eager_runner
        self.prefill_cuda_graph_runner = capture.prefill_runner
        self.decode_cuda_graph_runner = capture.decode.runner
        self.graph_mem_usage = capture.decode.graph_mem_usage

    def init_routed_experts_capturer(self):
        if self.is_draft_worker:
            # Capture is target-only. The draft worker runs in the same process
            # as its target and inits after it, so installing a capturer here
            # would overwrite the target's process-global one.
            return

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                model=self.model,
                model_config=self.model_config,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        set_global_indexer_capturer(
            create_indexer_capturer(
                model_config=self.model_config,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            maybe_downgrade_dtype_for_legacy_gpu(
                server_args=self.server_args, model_config=self.model_config
            )

        set_cuda_arch()

        self.load_config = self._build_load_config()
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.ps.tp_size
            )

        maybe_trigger_remote_instance_nccl_send_group(
            server_args=self.server_args, tp_rank=self.ps.tp_rank
        )

        self._load_model_with_memory_saver()

        if not self.is_draft_worker:
            get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        load_kv_cache_scales(model=self.model, server_args=self.server_args)

        self.sliding_window_size = resolve_sliding_window_size(
            self.model, self.model_config
        )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )

        report_online_quantization(model=self.model, server_args=self.server_args)

        maybe_register_debug_tensor_dump_hook(
            model=self.model,
            server_args=self.server_args,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=self.is_draft_worker,
            tp_size=self.ps.tp_size,
            tp_rank=self.ps.tp_rank,
            pp_rank=self.ps.pp_rank,
        )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        self._dist_barrier_after_load()

    def _build_load_config(self) -> LoadConfig:
        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        return LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.ps.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_weight_transport.engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_weight_transport.session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )

    def _load_model_with_memory_saver(self) -> None:
        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                self.remote_instance_weight_transport.weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        # Cache needs to be cleared after loading model weights (in the self.loader.load_model function).
        # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
        if _is_npu:
            torch.npu.empty_cache()
        monkey_patch_vllm_parallel_state(reverse=True)

    def _dist_barrier_after_load(self) -> None:
        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.ps.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def maybe_recover_ep_ranks(self):
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0
            self.eplb_manager.reset_generator()
            broadcast_global_expert_location_metadata(
                src_rank=get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.ps.tp_size,
            tp_rank=self.ps.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )
        if not cuda_graph_fully_disabled():
            init_lora_cuda_graph_moe_buffers(
                server_args=self.server_args,
                model=self.model,
                lora_manager=self.lora_manager,
                dtype=self.dtype,
            )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""
        return self.lora_manager.load_lora_adapter(lora_ref)

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        return self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""
        return self.lora_manager.unload_lora_adapter(lora_ref)

    def _get_attention_backend(self, init_new_workspace: bool = False):
        return get_attention_backend(
            model_runner=self, init_new_workspace=init_new_workspace
        )

    def init_decode_cuda_graph(self):
        capture = capture_decode_graph(model_runner=self)
        self.decode_cuda_graph_runner = capture.runner
        self.graph_mem_usage = capture.graph_mem_usage

    def init_prefill_cuda_graph(self, force_for_draft_worker: bool = False):
        self.prefill_cuda_graph_runner = capture_prefill_graph(
            model_runner=self,
            eager_runner=self.eager_runner,
            force_for_draft_worker=force_for_draft_worker,
        )

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def _prepare_eager_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Pad / normalize a batch for the eager (non-cuda-graph) forward.

        Runs the DP/MLP-sync padding, the attn-tp num_token_non_padded
        normalization, and the hisparse-coordinator refresh that the eager
        forward path needs — the cuda-graph path does the equivalent inside the
        runner's capture/replay, so this is skipped there.
        """
        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)

        # Normalize num_token_non_padded to be local to this attention TP rank if needed.
        # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
        # flavors already feed a zigzag-split rank-local layout whose token count
        # should not be further divided by attn_tp_size. MHA-arch prefill CP
        # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
        # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
        if (
            forward_batch.num_token_non_padded is not None
            and forward_batch.global_num_tokens_gpu is not None
            and require_gathered_buffer(self.server_args)
            and not is_dsa_enable_prefill_cp()
            and not is_mla_prefill_cp_enabled()
        ):
            forward_batch.adjust_num_token_non_padded_for_attn_tp(
                server_args=self.server_args,
            )

        # Hisparse coordinator — backends now read it from self.model_runner.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

    def _pp_kwargs(self, pp_proxy_tensors) -> dict:
        """Build the pp_proxy_tensors forward kwarg, in one place.

        Pipeline-parallel proxy tensors are threaded into model.forward only
        when the model accepts them (``support_pp``).
        """
        return {"pp_proxy_tensors": pp_proxy_tensors} if self.support_pp else {}

    def _extend_forward_kwargs(
        self, forward_batch: ForwardBatch, pp_proxy_tensors
    ) -> dict:
        """Build the extend/prefill model.forward kwargs (pp_proxy_tensors +
        input_embeds / replace_embeds overrides + get_embedding), shared by the
        prefill cuda-graph path and the EagerRunner's eager extend path."""
        kwargs = self._pp_kwargs(pp_proxy_tensors)
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True
        return kwargs

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        self.forward_pass_id += 1

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id
                if self.ps.dp_size is not None and self.ps.dp_size > 1
                else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = profile_range(build_step_span_name(forward_batch))

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (
            not self.is_draft_worker
            and (experts_capturer := get_global_experts_capturer()) is not None
        ):
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_recover_ep_ranks()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        with ctx_mgr:
            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
            )

            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                forward_batch.hisparse_coordinator = self.hisparse_coordinator
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Replay cuda graph if applicable
            if can_run_graph:
                ret = self.decode_cuda_graph_runner.execute(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # DP / MLP-sync padding + attn-tp normalization. Only the decode
            # cuda-graph path above pre-pads its static buffers and returns
            # early; split prefill, the prefill cuda graph, and the eager
            # forward all run the live batch and need this first — it sets
            # global_dp_buffer_len / padded token counts that graph eligibility
            # and the collectives depend on.
            self._prepare_eager_forward_batch(forward_batch)

            if forward_batch.forward_mode.is_split_prefill():
                # Layer-split mode; stays on ModelRunner, not the eager runner.
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif (
                forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
                and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)
                and self.prefill_cuda_graph_runner is not None
                and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
                and get_cp_strategy() is None
            ):
                category = (
                    "target_verify"
                    if forward_batch.forward_mode.is_target_verify()
                    else "extend"
                )
                # Prefill cuda graph (piecewise).
                kwargs = self._extend_forward_kwargs(forward_batch, pp_proxy_tensors)
                # TODO: device_timer.wrap is too broad here — it also includes
                # load_batch time. Move timing into the prefill cuda graph runner
                # to capture only the model.forward part.
                ctx = (
                    self.device_timer.wrap(metadata={"category": category})
                    if self.device_timer
                    else contextlib.nullcontext()
                )
                with ctx:
                    ret = self.prefill_cuda_graph_runner.execute(
                        forward_batch, **kwargs
                    )
                can_run_graph = True
            else:
                # Eager: decode / extend / idle dispatched inside the runner.
                ret = self.eager_runner.execute(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.ngram_embedding_manager.update_after_decode(
            next_token_ids=next_token_ids,
            forward_batch=forward_batch,
        )
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    def check_weights(self, action: str):
        return self._weight_checker.handle(action=action)

    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and not elastic_ep_state.is_active_equal_last():
            elastic_ep_state.snapshot_active_to_last()
            elastic_ep_state.sync_active_to_cpu()
            logging.info("EPLB due to rank faults")
            gen = self.eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output

    def update_model_fields(
        self,
        new_model: torch.nn.Module,
        *,
        model_path: str,
        load_format: str,
        load_config: LoadConfig,
    ) -> None:
        """Commit a newly-loaded model and its provenance (post-load hook)."""
        self.model = new_model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config
