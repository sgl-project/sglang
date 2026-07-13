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
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
    dsa_layer_skips_topk,
    get_num_indexer_layers,
    is_deepseek_dsa,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.distributed import (
    bootstrap,
    get_tp_group,
    get_world_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    prealloc_symmetric_memory_pool,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.dllm.config import DllmConfig
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
    format_expert_location_layout,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.eplb.lplb_solver import (
    LPLBSolver,
    assert_lplb_supported_model,
    clear_global_lplb_solvers,
    set_global_lplb_solver,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.hardware_backend.xpu.graph_runner.xpu_graph_runner import XPUGraphRunner
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper, model_parallel
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.cp.utils import (
    get_cp_strategy,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager, init_lora_cuda_graph_moe_buffers
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache import kv_cache_dtype
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
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
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_kv_cache_scales,
    maybe_downgrade_dtype_for_legacy_gpu,
    maybe_register_debug_tensor_dump_hook,
    maybe_trigger_remote_instance_nccl_send_group,
    report_online_quantization,
    resolve_sliding_window_size,
)
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
    NgramEmbeddingManager,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_executor.runner import (
    EagerRunner,
    PrefillCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_flags, get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (  # noqa: F401  (re-export)
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    ServerArgs,
    add_chunked_prefix_cache_attention_backend,
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
    get_bool_env_var,
    init_cublas,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    numa_utils,
    require_gathered_buffer,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.network import get_local_ip_auto
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

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

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


class ModelRunner(ModelRunnerKVCacheMixin):
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
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = server_args.dcp_size
        self.dcp_rank = self.tp_rank % self.dcp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.attn_dp_size = (
            server_args.dp_size if server_args.enable_dp_attention else 1
        )
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
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
        self.capture_tail_hooks = []
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = model_config.is_hybrid_swa_compress
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.init_remote_instance_weight_transporter()

        self.msprobe_debugger = None
        if server_args.msprobe_dump_config is not None:
            self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.eagle_use_aux_hidden_state = False
        self.eagle_draft_num_layers = None
        self.dflash_family_use_aux_hidden_state = False
        self.dflash_family_target_layer_ids = None
        self.dflash_family_draft_num_layers = None
        if (
            (self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone())
            and not self.is_draft_worker
            and server_args.speculative_draft_model_path
        ):
            # Load draft config to get layer count for KV cache sizing
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=server_args.speculative_draft_model_path,
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            num_nextn_predict_layers = draft_model_config.num_nextn_predict_layers
            if num_nextn_predict_layers is not None:
                self.eagle_draft_num_layers = int(num_nextn_predict_layers)
            else:
                self.eagle_draft_num_layers = int(
                    max(
                        draft_model_config.num_hidden_layers,
                        draft_model_config.num_attention_layers,
                    )
                )

            if self.spec_algorithm.is_eagle3():
                self.eagle_use_aux_hidden_state = True
                try:
                    eagle_config = getattr(
                        draft_model_config.hf_config, "eagle_config", None
                    )
                    self.eagle_use_aux_hidden_state = eagle_config.get(
                        "use_aux_hidden_state", True
                    )
                    self.eagle_aux_hidden_state_layer_ids = eagle_config[
                        "eagle_aux_hidden_state_layer_ids"
                    ]
                except:
                    # if there is no aux layer, set to None
                    self.eagle_aux_hidden_state_layer_ids = None

        if self.spec_algorithm.is_dflash_family() and not self.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

            # Select target layers to capture for building draft context features.
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            dflash_draft_config = parse_dflash_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            draft_num_layers = dflash_draft_config.require_num_layers()
            trained_target_layers = dflash_draft_config.num_target_layers

            target_num_layers = getattr(
                self.model_config.hf_text_config, "num_hidden_layers", None
            )
            if target_num_layers is None:
                raise ValueError(
                    "Block-draft-with-target-kv spec requires target num_hidden_layers "
                    f"in config. Got target={target_num_layers}."
                )
            target_num_layers = int(target_num_layers)

            if (
                trained_target_layers is not None
                and trained_target_layers != target_num_layers
            ):
                logger.warning(
                    "Draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                    "selecting capture layers based on the runtime target model.",
                    trained_target_layers,
                    target_num_layers,
                )

            target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
                target_num_layers=int(target_num_layers),
                draft_num_layers=int(draft_num_layers),
            )

            if self.spec_algorithm.is_dspark():
                from sglang.srt.speculative.dspark_components.dspark_config import (
                    parse_dspark_draft_config,
                )

                dspark_draft_config = parse_dspark_draft_config(
                    draft_hf_config=draft_model_config.hf_config
                )
                if not dspark_draft_config.require_markov():
                    raise ValueError(
                        "DSPARK requires markov_rank > 0 in the draft config, "
                        f"got markov_rank={dspark_draft_config.markov_rank}."
                    )
                if dspark_draft_config.target_layer_ids is not None:
                    target_layer_ids = list(dspark_draft_config.target_layer_ids)

            self.dflash_family_use_aux_hidden_state = True
            self.dflash_family_draft_num_layers = int(draft_num_layers)
            self.dflash_family_target_layer_ids = target_layer_ids

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Chunked prefix caching requires an MLA model on a backend whose
        # kernels read that layout. This is a load-time gate, not a
        # resolution-time one: out-of-tree platforms register their supported
        # backends in init_backend(), which runs when this module is imported
        # — after ServerArgs.__post_init__. Target runner only: a draft
        # model's (often non-MLA) config must not flip the shared setting.
        if not self.is_draft_worker and (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            if not server_args.disable_chunked_prefix_cache:
                server_args.override(
                    "model_runner.chunked_prefix_cache_gate",
                    disable_chunked_prefix_cache=True,
                )
        if not self.is_draft_worker and not server_args.disable_chunked_prefix_cache:
            logger.info("Chunked prefix cache is turned on.")

        # Set the global server_args in the scheduler process (target worker
        # only, so a draft init cannot clobber target-derived global state).
        if not self.is_draft_worker:
            set_global_server_args_for_scheduler(server_args)

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Set float32 matmul precision
        if get_server_args().enable_tf32_matmul:
            torch.set_float32_matmul_precision("high")

        # Get available memory before model loading.
        # Stored for later use by alloc_memory_pool().
        self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

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
        self.check_quantized_moe_compatibility()

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
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

        if self.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self.init_weight_updater()
        self.init_weight_exporter()

    def init_weight_updater(self):
        self.weight_updater = WeightUpdater(
            tp_rank=self.tp_rank,
            device=self.device,
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            custom_weight_loaders=self.server_args.custom_weight_loader,
            get_model=lambda: self.model,
            update_model_fields=self.update_model_fields,
            recapture_cuda_graph=self.init_decode_cuda_graph,
            get_model_runner=lambda: self,
        )

    def init_weight_exporter(self):
        self.weight_exporter = WeightExporter(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            gpu_id=self.gpu_id,
            get_model_path=lambda: self.model_config.model_path,
            get_model=lambda: self.model,
        )

    def init_remote_instance_weight_transporter(self):
        self.remote_instance_weight_transporter = RemoteInstanceWeightTransporter(
            server_args=self.server_args,
            get_model=lambda: self.model,
            tp_rank=self.tp_rank,
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

    def init_msprobe(self):
        # Init the msprobe
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError:
            logger.warning(
                "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
                "see https://gitcode.com/Ascend/msprobe for details."
            )
            return
        seed_all(mode=True)
        self.msprobe_debugger = PrecisionDebugger(
            config_path=self.server_args.msprobe_dump_config
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
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
            self.remote_instance_weight_transporter.init_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    "Initial expert_location_metadata:\n%s",
                    format_expert_location_layout(
                        get_global_expert_location_metadata()
                    ),
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        if self.server_args.ep_dispatch_algorithm == "lp" and not self.is_draft_worker:
            self._init_lplb_solvers()

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
        self._prepare_moe_topk()

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

        self.remote_instance_weight_transporter.maybe_register_and_publish_weight_info()

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        # Some EAGLE3 drafts (e.g. nvidia/Kimi-K2.5-Thinking-Eagle3) carry the full DeepSeek-V3
        # config schema and explicitly set `num_nextn_predict_layers: 0`. Treat that the same as
        # the field being absent — otherwise the draft worker takes the MTP branch below with
        # model_num_layers=0, sizing the draft KV pool to zero and producing an IndexError on
        # the first forward (`set_mla_kv_buffer` -> `self.kv_buffer[layer_id - self.start_layer]`).
        _nnpl = self.model_config.num_nextn_predict_layers
        model_has_mtp_layers = _nnpl is not None and _nnpl > 0
        if self.is_draft_worker and model_has_mtp_layers:
            model_num_layers = getattr(
                self.model, "num_stages", self.model_config.num_nextn_predict_layers
            )
        else:
            model_num_layers = max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        self.adjust_hybrid_swa_layers_for_pp()

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(self.model, get_server_args().torchao_config)

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        self.configure_kv_cache_dtype()

    def get_pp_proxy_topk_size(self) -> Optional[int]:
        hf_config = self.model_config.hf_text_config
        if (
            self.pp_size <= 1
            or self.pp_rank == 0
            or not is_deepseek_dsa(hf_config)
            or not dsa_layer_skips_topk(hf_config, self.start_layer)
        ):
            return None
        return getattr(hf_config, "index_topk", None)

    def decode_num_tokens_per_req(
        self, *, num_draft_tokens: Optional[int] = None
    ) -> int:
        """Logits rows per decode batch slot."""
        if self.spec_algorithm.is_speculative():
            if num_draft_tokens is None:
                num_draft_tokens = self.server_args.speculative_num_draft_tokens
            return self.spec_algorithm.get_num_tokens_per_req_for_target_verify(
                num_draft_tokens, self.is_draft_worker
            )
        dllm_config = DllmConfig.from_server_args(self.server_args)
        return dllm_config.block_size if dllm_config is not None else 1

    def max_decode_logits_rows(self) -> int:
        """Rows the shared logits buffer needs."""
        num_tokens_per_req = self.decode_num_tokens_per_req()
        capture_bs, _ = get_batch_sizes_to_capture(self, num_tokens_per_req)
        return max(capture_bs) * num_tokens_per_req

    def alloc_memory_pool(self, memory_pool_config: Optional[MemoryPoolConfig] = None):
        """Allocate KV cache memory pools only (no backends or cuda graphs)."""
        if memory_pool_config is not None:
            self.memory_pool_config = memory_pool_config

        self.init_memory_pool(self.pre_model_load_memory)

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
                swap_in_block_size=hisparse_cfg.swap_in_block_size,
            )

        self.init_routed_experts_capturer()
        self.init_indexer_capturer()

        self.graph_shared_output = None

    def init_attention_backends(self):
        """Initialize attention backends only (no cuda graph capture)."""
        # TODO: Refactor device-specific init branches into platform interface (separate PR).
        # Must be called BEFORE init_decode_cuda_graph() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        self.init_aux_hidden_state_capture()

        if self.device == "cuda" or self.device == "musa":
            init_cublas()
            self.init_attention_backend()
        elif self.device in ["cpu", "xpu"]:
            self.init_attention_backend()
        elif self.device == "npu":
            self.init_attention_backend()
            # lazy init for zbal with mix mode (before graph capture when enable_cuda_graph)
            if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 and not self.is_draft_worker:
                from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

                lazy_init_zbal_gva_mem(
                    self.device,
                    self.gpu_id,
                    get_world_group().rank_in_group,
                    get_world_group().world_size,
                    get_world_group().cpu_group,
                )
        else:
            self.init_attention_backend()

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        """Capture cuda graphs. Requires init_attention_backends() to have run.

        Spec draft runners pass capture_decode_cuda_graph=False
        because they capture their own decode-style graphs separately.
        """

        self.graph_shared_output = GraphSharedOutput.create_for_model_runner(self)

        # The eager (no-cuda-graph) phase runner, built AFTER the attention
        # backend so its __init__ can warm up kernels (run-once) and allocate the
        # fixed-max static buffer — both before the cuda-graph runners, so that
        # buffer is canonical in the shared pool and the cg runners coalesce onto
        # it. Always built: it serves both the fully-disabled case (decode/prefill
        # runners point at it) and the eager fallback when a cg runner can't run a
        # batch.
        self.eager_runner = EagerRunner(self)

        # cuda-graph capture: prefill before decode, so both coalesce onto the
        # eager buffer allocated above. (init_prefill_cuda_graph routes prefill
        # to the eager runner when the prefill graph is disabled.)
        self.init_prefill_cuda_graph()

        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0

        if capture_decode_cuda_graph:
            if self.device in ("cuda", "musa", "cpu", "npu", "xpu"):
                self.init_decode_cuda_graph()
            elif (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            ):
                self.init_decode_cuda_graph()
        else:
            self.decode_cuda_graph_runner = self.eager_runner

        # Register forward hooks AFTER cuda-graph capture so their tensor ops are
        # not traced into any captured graph — capture stays hook-free and hooks
        # fire only on the eager forward path (capture replay never runs Python
        # hooks anyway).
        if self.server_args.forward_hooks:
            register_forward_hooks(self.model, self.server_args.forward_hooks)

        prealloc_symmetric_memory_pool(
            is_draft_worker=self.is_draft_worker,
            enable_symm_mem=self.server_args.enable_symm_mem,
            device=self.device,
            forward_stream=self.forward_stream,
        )

        if self.canary_manager is not None and not self.is_draft_worker:
            self.canary_manager.mark_init_finished()

    def adjust_hybrid_swa_layers_for_pp(self):
        if not self.is_hybrid_swa:
            return

        if self.model_config.is_deepseek_v4_arch:
            return

        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

    def init_routed_experts_capturer(self):
        if self.is_draft_worker:
            # Capture is target-only. The draft worker runs in the same process
            # as its target and inits after it, so installing a capturer here
            # would overwrite the target's process-global one.
            return

        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        enable = get_server_args().enable_return_indexer_topk
        # Producer wiring is CUDA-only (Indexer.forward_cuda + MLA skip_topk
        # path); other backends would create a capturer but never feed it.
        if enable and self.device != "cuda":
            logger.warning(
                "indexer-topk capture is CUDA-only; %s backend not yet wired. "
                "Disabling capturer.",
                self.device,
            )
            set_global_indexer_capturer(None)
            return

        hf_text_config = self.model_config.hf_text_config
        num_indexer_layers = get_num_indexer_layers(hf_text_config)
        index_topk = getattr(hf_text_config, "index_topk", 0)
        set_global_indexer_capturer(
            create_indexer_capturer(
                enable=enable,
                num_indexer_layers=num_indexer_layers,
                index_topk=index_topk,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_aux_hidden_state_capture(self):
        """Configure auxiliary hidden state capture for speculative decoding.

        Must be called before CUDA graph capture so the captured graphs
        include aux hidden state output paths.
        """
        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )
        if self.dflash_family_use_aux_hidden_state:
            if self.spec_algorithm.is_dspark() and hasattr(
                self.model, "set_dspark_layers_to_capture"
            ):
                self.model.set_dspark_layers_to_capture(
                    self.dflash_family_target_layer_ids
                )
            elif hasattr(self.model, "set_dflash_layers_to_capture"):
                self.model.set_dflash_layers_to_capture(
                    self.dflash_family_target_layer_ids
                )
            else:
                raise ValueError(
                    f"Model {self.model.__class__.__name__} implements neither "
                    "set_dspark_layers_to_capture nor set_dflash_layers_to_capture, "
                    "one of which is required for DFLASH/DSPARK."
                )

    def check_quantized_moe_compatibility(self):
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size // self.moe_dp_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (
                not envs.SGLANG_SHARED_EXPERT_TP1.get()
                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
                and not _use_aiter
            ):
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        result = bootstrap.init_torch_distributed(
            server_args=self.server_args,
            model_config=self.model_config,
            device=self.device,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            dp_size=self.attn_dp_size,
            attn_cp_size=self.attn_cp_size,
            moe_ep_size=self.moe_ep_size,
            moe_dp_size=self.moe_dp_size,
            dcp_size=self.dcp_size,
            dist_port=self.dist_port,
            is_draft_worker=self.is_draft_worker,
            local_omp_cpuid=self.local_omp_cpuid if self.device == "cpu" else None,
        )
        self.tp_group = result.tp_group
        self.pp_group = result.pp_group
        self.attention_tp_group = result.attention_tp_group
        self.pre_model_load_memory = result.pre_model_load_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
                and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
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
                self.model_config, self.load_config, self.tp_size
            )

        maybe_trigger_remote_instance_nccl_send_group(
            server_args=self.server_args, tp_rank=self.tp_rank
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

        self.prefill_aware_swa = (
            hasattr(self.model, "is_prefill_aware_swa")
            and self.model.is_prefill_aware_swa()
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
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
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

    def _prepare_moe_topk(self):
        balancer_cls = None
        num_prepared = 0
        num_routed_experts = None
        for module in self.model.modules():
            if not isinstance(module, (TopK, HashTopK)):
                continue
            if not module.enable_waterfill or module.waterfill_balancer is not None:
                continue
            if num_routed_experts is None:
                num_routed_experts = getattr(
                    self.model_config.hf_config, "n_routed_experts", None
                )
                if num_routed_experts is None:
                    raise ValueError(
                        "Waterfill requires model config n_routed_experts."
                    )
            if balancer_cls is None:
                from sglang.srt.layers.moe.waterfill import WaterfillBalancer

                balancer_cls = WaterfillBalancer
            # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
            # Redundant experts therefore need to be included in the per-rank
            # expert count used for Waterfill's shared-expert slot remapping.
            num_physical_routed_experts = (
                num_routed_experts + self.server_args.ep_num_redundant_experts
            )
            if isinstance(module, TopK):
                routed_scaling_factor = module.topk_config.routed_scaling_factor
            else:
                routed_scaling_factor = module.routed_scaling_factor
            module.waterfill_balancer = balancer_cls(
                num_routed_experts=num_physical_routed_experts,
                world_size=self.moe_ep_size,
                rank=self.moe_ep_rank,
                layer_id=module.layer_id,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
            )
            num_prepared += 1
        if num_prepared:
            log_info_on_rank0(
                logger, f"Prepared {num_prepared} Waterfill TopK modules."
            )

    def _init_lplb_solvers(self):
        """Initialize per-layer LPLB solvers from current expert location metadata."""
        from sglang.srt.distributed import get_moe_ep_group

        # Gate: refuse LP for non-DeepSeek MoE families whose empty-token paths
        # don't participate in the EP all-reduce (would deadlock under DP-
        # attention). Failure here happens before any forward pass.
        architectures = getattr(self.model_config.hf_config, "architectures", None)
        if architectures:
            assert_lplb_supported_model(architectures[0])

        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return
        clear_global_lplb_solvers()
        ep_group = get_moe_ep_group()
        for lid in range(metadata.num_layers):
            solver = LPLBSolver(
                phy2log=metadata.physical_to_logical_map[lid],
                log2phy=metadata.logical_to_all_physical_map[lid],
                num_gpus=metadata.ep_size,
                ep_group=ep_group,
                logical_to_all_physical_map_num_valid=(
                    metadata.logical_to_all_physical_map_num_valid[lid]
                ),
            )
            set_global_lplb_solver(lid, solver)
        logger.info(f"Initialized LPLB solvers for {metadata.num_layers} layers")

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
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_weight_transporter.engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_weight_transporter.session_id,
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
                self.remote_instance_weight_transporter.weight_info = (
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
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
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
                src_rank=self._get_healthy_expert_location_src_rank(
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

    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
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

    @property
    def effective_max_total_num_tokens(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            return self.full_max_total_num_tokens or self.swa_max_total_num_tokens
        else:
            return self.max_total_num_tokens

    def _record_kv_cache_dtype(self, resolved: str) -> None:
        # Load-time resolution transition: the weight-resolved kv-cache dtype
        # is declared into the flags tier; the dual-apply inside the helper
        # replaces the legacy in-place write. Mock runners whose server_args
        # is not the published object keep the plain write.
        from sglang.srt.runtime_context import get_context

        if get_context()._server_args is self.server_args:
            from sglang.srt.arg_groups.overrides import declare_load_time_override

            declare_load_time_override(
                "ModelRunner.configure_kv_cache_dtype",
                {"kv_cache_dtype": resolved},
            )
        else:
            self.server_args.override(
                "ModelRunner.configure_kv_cache_dtype", kv_cache_dtype=resolved
            )

    def configure_kv_cache_dtype(self):
        resolved_kv_cache_dtype, self.kv_cache_dtype = (
            kv_cache_dtype.configure_kv_cache_dtype(
                server_args_kv_cache_dtype=self.server_args.kv_cache_dtype,
                model=self.model,
                model_dtype=self.dtype,
                is_draft_worker=self.is_draft_worker,
                is_dflash=self.spec_algorithm.is_dflash(),
                speculative_draft_attention_backend=self.server_args.speculative_draft_attention_backend,
            )
        )
        if resolved_kv_cache_dtype is not None:
            self._record_kv_cache_dtype(resolved_kv_cache_dtype)

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

        # Record resolved per-mode backends on the backend for model dispatch.
        self.attn_backend.prefill_attention_backend_str = (
            self.prefill_attention_backend_str
        )
        self.attn_backend.decode_attention_backend_str = (
            self.decode_attention_backend_str
        )

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            # Single backend for all draft modes (no prefill/decode split).
            self.prefill_attention_backend_str = draft_attn_backend
            self.decode_attention_backend_str = draft_attn_backend
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def init_decode_cuda_graph(self):
        """Capture device graphs."""
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        ):
            return

        if self.device == "cpu" and not get_flags().capture.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: f"{current_platform.device_name} graph",
            {
                "cuda": "CUDA graph",
                "musa": "CUDA graph",
                "cpu": "CPU graph",
                "npu": "NPU graph",
                "xpu": "XPU graph",
            },
        )
        role = "draft" if self.is_draft_worker else "target"
        if self.spec_algorithm.is_speculative():
            capture_name = f"{role} verify"
            num_tokens_per_req = (
                self.spec_algorithm.get_num_tokens_per_req_for_target_verify(
                    self.server_args.speculative_num_draft_tokens,
                    self.is_draft_worker,
                )
            )
        else:
            capture_name = f"{role} decode"
            num_tokens_per_req = 1
        capture_bs, _ = get_batch_sizes_to_capture(self, num_tokens_per_req)
        decode_backend = self.server_args.cuda_graph_config.decode.backend
        logger.info(
            f"Capture {capture_name} {graph_backend[self.device]} begin. "
            f"backend={decode_backend}, num_tokens_per_req={num_tokens_per_req}, "
            f"bs={capture_bs}, avail mem={before_mem:.2f} GB"
        )

        if current_platform.is_out_of_tree():
            GraphRunnerCls = current_platform.get_graph_runner_cls()
            self.decode_cuda_graph_runner = GraphRunnerCls(self)
        else:
            from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
                DecodeCudaGraphRunner,
            )

            graph_runners = defaultdict(
                lambda: DecodeCudaGraphRunner,
                {
                    "cpu": CPUGraphRunner,
                    "npu": NPUGraphRunner,
                    "xpu": XPUGraphRunner,
                },
            )
            self.decode_cuda_graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {capture_name} {graph_backend[self.device]} end. "
            f"elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={self.graph_mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
        )

    def init_prefill_cuda_graph(self, force_for_draft_worker: bool = False):
        """Initialize prefill CUDA graph runner."""
        self.prefill_cuda_graph_runner = None

        if check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
            logger.info(
                "Disable prefill CUDA graph because cuda_graph_config "
                "resolved prefill.backend='disabled' (e.g. via "
                "--cuda-graph-backend-prefill=disabled or auto-disable rules)."
            )
            # Prefill cuda graph disabled: route eager prefill through the
            # EagerRunner (its can_run_graph returns False, so _forward_raw's
            # extend branch falls through to the eager path).
            if not self.is_draft_worker:
                self.prefill_cuda_graph_runner = self.eager_runner
            return

        # Draft models skip here during __init__; the eagle worker calls
        # this method explicitly (force_for_draft_worker=True) after
        # init_lm_head so graphs capture the final embedding weights.
        if self.is_draft_worker and not force_for_draft_worker:
            return

        # Skip prefill CG for EAGLE target on tc_piecewise: that backend
        # captures CaptureHiddenMode.NULL while runtime requests FULL, so
        # the captured graph is dead, and capturing it perturbs FP4 /
        # TRTLLM-MoE state and corrupts decode replay (see #28386). BCG
        # captures FULL for EAGLE target in PrefillCudaGraphRunner.__init__
        # (restored from #25795), so it does NOT need this skip.
        if (
            self.spec_algorithm.is_eagle()
            and not self.is_draft_worker
            and not self.server_args.enable_return_hidden_states
            and not check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
        ):
            logger.info(
                "Disable prefill CUDA graph for EAGLE target on tc_piecewise "
                "to avoid FP4/MoE decode-replay corruption (#28386)."
            )
            self.prefill_cuda_graph_runner = self.eager_runner
            return

        # Resolve the decoder once. Some VLM wrappers (for example Kimi-VL)
        # expose it as ``language_model`` rather than ``model``.
        try:
            language_model = resolve_language_model(self.model)
        except AttributeError:
            logger.warning(
                "Disable prefill CUDA graph because the model is not a language model"
            )
            return

        # Disable prefill CUDA graph for non capture size
        if not self.server_args.cuda_graph_config.prefill.bs:
            logger.warning(
                "Disable prefill CUDA graph because the capture size is not set"
            )
            return

        # Collect attention layers and moe layers from the model. Keep a VLM
        # wrapper that exposes ``language_model`` unchanged: assigning it to
        # ``model`` would register a duplicate module alias and duplicate the
        # model's state-dict namespace.
        if hasattr(self.model, "model"):
            self.model.model = language_model

        # Find the module that owns the decoder `layers`. Models wrap it at
        # varying depths: a direct text model exposes `.layers`, a CausalLM
        # wraps it as `.model.layers`, and some multimodal models add another
        # level (e.g. DeepSeek-OCR: OCR wrapper -> Deepseek*ForCausalLM ->
        # text model -> `.layers`). Descend the `.model` chain until we find it.
        layer_model = language_model
        while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
            layer_model = layer_model.model

        if not hasattr(layer_model, "layers"):
            logger.warning(
                "Disable prefill CUDA graph because the model does not have a 'layers' attribute"
            )
            return

        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        self.dsa_indexers = []
        for layer in layer_model.layers:
            attn_layer = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_layer = layer.self_attn.attn
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    attn_layer = layer.self_attn.attn_mqa
                    if _is_hip and hasattr(layer.self_attn, "attn_mha"):
                        attn_layer._pcg_mha_companion = layer.self_attn.attn_mha
            # For hybrid model
            elif hasattr(layer, "attn"):
                attn_layer = layer.attn
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    attn_layer = layer.linear_attn.attn
                else:
                    attn_layer = layer.linear_attn
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attn_layer = layer.attention.attn
            # For NemotronH and similar hybrid models using 'mixer' attribute
            elif hasattr(layer, "mixer"):
                if hasattr(layer.mixer, "attn"):
                    attn_layer = layer.mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    # Mamba layer with split op support - store the layer itself
                    attn_layer = layer

            if attn_layer is not None:
                self.attention_layers.append(attn_layer)
            elif hasattr(layer, "mixer"):
                self.attention_layers.append(None)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            # For NemotronH MoE layers using 'mixer' attribute
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
                moe_block = layer.mixer.experts
                moe_fusion = layer.mixer
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)
            # NSA indexers (None for layers without NSA)
            dsa_indexer = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "indexer"):
                dsa_indexer = layer.self_attn.indexer
            self.dsa_indexers.append(dsa_indexer)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            log_info_on_rank0(
                logger,
                "Disable prefill CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        prefill_backend = self.server_args.cuda_graph_config.prefill.backend
        role = "draft" if self.is_draft_worker else "target"
        capture_name = f"{role} prefill"
        capture_num_tokens = sorted(self.server_args.cuda_graph_config.prefill.bs)
        logger.info(
            f"Capture {capture_name} CUDA graph begin. "
            f"backend={prefill_backend}, num_tokens={capture_num_tokens}, "
            f"avail mem={before_mem:.2f} GB"
        )

        self.prefill_cuda_graph_runner = PrefillCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {capture_name} CUDA graph end. "
            f"elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        self.local_omp_cpuid = numa_utils.init_threads_binding(
            tp_rank=self.tp_rank, tp_size=self.tp_size
        )

    def apply_torch_tp(self):
        model_parallel.apply_torch_tp(
            model=self.model, device=self.device, tp_size=self.tp_size
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
                if self.attn_dp_size is not None and self.attn_dp_size > 1
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

        if self.enable_elastic_ep:
            self.maybe_recover_ep_ranks()

        return output

    def _maybe_execute_deferred_mamba_cow_and_clear(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Run deferred clear/COW on the forward stream, before the mamba layers
        read the pool, so the copies don't race the scheduler copy stream.

        No-op unless this is an extend forward on a mamba model's target worker;
        COW/clear only happen at prefix match on extend.
        """
        pool = self.req_to_token_pool
        if (
            not isinstance(pool, HybridReqToTokenPool)
            or self.is_draft_worker
            or not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            # mamba_pool is a pure PHYSICAL store; translate before zeroing or
            # clear_slots zeroes the wrong physical slots.
            pool.mamba_pool.clear_slots(
                pool.translate_mamba_indices(forward_batch.mamba_clear_indices)
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            if pool.mamba_ckpt_pool is not None:
                # int8 checkpoints: dequantize src int8 ckpt slot into the active bf16 dst.
                pool.mamba_ckpt_pool.load_to_active(
                    pool.mamba_pool,
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
            else:
                # mamba_pool is a pure PHYSICAL store; translate both COW slot ids.
                pool.mamba_pool.copy_from(
                    pool.translate_mamba_indices(forward_batch.mamba_cow_src_indices),
                    pool.translate_mamba_indices(forward_batch.mamba_cow_dst_indices),
                )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

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

            # Deferred mamba COW/clear on the forward stream, before the extend
            # dispatch below reads the pool.
            self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

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

    def check_weights(self, action: str, allow_quant_error: bool = False):
        return self._weight_checker.handle(
            action=action, allow_quant_error=allow_quant_error
        )

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
        self.model = new_model
        self.server_args.override(
            "model_runner.update_model_fields",
            model_path=model_path,
            load_format=load_format,
        )
        self.load_config = load_config
