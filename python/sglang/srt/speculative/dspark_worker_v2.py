import contextlib
import logging
import os
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_attn_tp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.dspark_info import (
    DSparkDraftBlockInput,
    DSparkDraftInputV2,
    DSparkVerifyInput,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.dspark import (
    _compute_dspark_accept_bonus_triton_unchecked,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import empty_context

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "off", "no")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_plan_stream(
    device: str,
) -> Tuple[object, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    return None, contextlib.nullcontext()


class DSparkWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        with (
            empty_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker = TpModelWorker(
                server_args=draft_server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                pp_rank=0,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                dp_rank=dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self._draft_worker.model_runner
        self._draft_worker.draft_runner = self.draft_model_runner
        self.draft_model = self.draft_model_runner.model
        self._draft_inner = self.draft_model.model

        has_own_embed_tokens = bool(
            getattr(self.draft_model, "has_own_embed_tokens", False)
        )
        has_own_lm_head = bool(getattr(self.draft_model, "has_own_lm_head", False))
        shared_embed_tokens = not has_own_embed_tokens
        shared_lm_head = not has_own_lm_head
        if shared_embed_tokens:
            self.draft_model.model.embed_tokens.weight = (
                self.target_worker.model_runner.model.model.embed_tokens.weight
            )
        if shared_lm_head:
            self.draft_model.lm_head.weight = (
                self.target_worker.model_runner.model.lm_head.weight
            )
        if self._is_tp0():
            logger.warning(
                "DSpark draft embed/head sharing: has_own_embed_tokens=%s "
                "has_own_lm_head=%s shared_embed_tokens=%s shared_lm_head=%s",
                has_own_embed_tokens,
                has_own_lm_head,
                shared_embed_tokens,
                shared_lm_head,
            )

        requested_block_size = int(server_args.speculative_num_draft_tokens)
        model_block_size = int(
            getattr(self.draft_model, "block_size", requested_block_size)
        )
        if self._is_tp0() and model_block_size != requested_block_size:
            logger.warning(
                "DSpark block size mismatch: using speculative_num_draft_tokens=%s "
                "but draft model block_size=%s.",
                requested_block_size,
                model_block_size,
            )
        # DeepSpec's block_size is the number of sampled draft tokens. SGLang's
        # speculative stride includes the anchor/bonus slot, so DSpark verifies
        # anchor + block_size sampled drafts.
        self.block_size = int(model_block_size)
        self.verify_stride = int(self.block_size) + 1
        self.speculative_num_draft_tokens = int(self.verify_stride)
        # DSpark uses different fixed widths for the two graphable paths:
        # draft-block runs anchor + (block_size - 1) noise query rows, while
        # target verify consumes anchor + block_size sampled candidates.
        self._draft_worker.server_args.speculative_num_draft_tokens = int(
            self.block_size
        )
        self.draft_model_runner.server_args.speculative_num_draft_tokens = int(
            self.block_size
        )

        self.noise_token_id = int(self._draft_inner.noise_token_id)
        self.markov_rank = int(self._draft_inner.markov_rank)
        self.num_dspark_layers = int(self.draft_model.num_dspark_layers)
        self.confidence_threshold = float(
            server_args.speculative_dspark_confidence_threshold
        )
        # vLLM's DSpark inference path does not wire confidence_head into
        # acceptance yet. Keep computing it for diagnostics, but do not let an
        # incompatible or uncalibrated confidence head truncate valid matches
        # unless explicitly requested.
        self._use_confidence_gate = _env_flag(
            "SGLANG_DSPARK_USE_CONFIDENCE_GATE", False
        )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._verify_pos_offsets = torch.arange(
            self.verify_stride, device=self.device, dtype=torch.int64
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        # The current Triton kernel assumes candidates and confidence have the
        # same width. DeepSpec DSpark has candidates=[anchor]+N drafts, while
        # confidence has N draft rows, so use the eager verifier for now.
        self._use_triton_accept_bonus = False
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []
        self._markov_refine_buffer_cap: int = 0
        self._markov_candidates_buf: Optional[torch.Tensor] = None
        self._markov_embeds_buf: Optional[torch.Tensor] = None
        self._vocab_shard_mapping_cache: dict[
            tuple[int, int, torch.device], torch.Tensor
        ] = {}
        self._last_markov_refine_debug: Optional[dict] = None
        self.draft_attn_backend = None
        self.draft_extend_attn_backend = None
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)
        self._accept_anomaly_enabled = _env_flag("SGLANG_DSPARK_DEBUG_ACCEPT", True)
        self._accept_anomaly_topk_enabled = _env_flag(
            "SGLANG_DSPARK_DEBUG_ACCEPT_TOPK", True
        )
        self._accept_anomaly_threshold = max(
            1, _env_int("SGLANG_DSPARK_DEBUG_ACCEPT_THRESHOLD", 8)
        )
        self._accept_anomaly_history_size = max(
            1, _env_int("SGLANG_DSPARK_DEBUG_ACCEPT_HISTORY", 8)
        )
        self._accept_anomaly_max_dumps = max(
            1, _env_int("SGLANG_DSPARK_DEBUG_ACCEPT_MAX_DUMPS", 64)
        )
        self._accept_anomaly_histories: dict[tuple[int, object], list[dict]] = {}
        self._accept_anomaly_streaks: dict[tuple[int, object], int] = {}
        self._accept_anomaly_dumped: set[tuple[int, object]] = set()
        self._accept_anomaly_dump_count = 0
        self._last_draft_block_metadata_debug: Optional[dict] = None
        self._last_draft_block_ids_debug: Optional[list[int]] = None
        self._last_draft_visible_block_swa_locs_debug: Optional[list[int]] = None
        self._last_context_kv_path_debug: Optional[str] = None
        self._boundary_debug_by_req_pool: dict[int, dict] = {}
        self._target_aux_debug_by_req_pool: dict[int, dict] = {}
        self._swa_write_source_by_req_pool: dict[int, dict[int, str]] = {}
        self._stacked_wqkv_fp8_proj = None
        self._stacked_wqkv_kv_offsets: list[tuple[int, int]] = []
        self._stacked_wqkv_out_sizes: list[int] = []
        self._init_fp8_wqkv_stack()

        if self._is_tp0():
            config = getattr(self._draft_inner, "config", None)
            target_layer_ids = [
                int(x) for x in list(getattr(self._draft_inner, "target_layer_ids", []))
            ]
            decoder_layer_ids = list(target_layer_ids)
            compress_ratios = list(getattr(config, "compress_ratios", []) or [])
            draft_config_compress_ratios = compress_ratios[
                int(getattr(config, "num_hidden_layers", 0)) :
            ]
            draft_runtime_compress_ratios = [
                int(getattr(layer.self_attn, "compress_ratio", -1))
                for layer in self._draft_inner.layers
            ]
            markov_head = self._draft_inner.markov_head
            markov_w1_shape = tuple(markov_head.markov_w1.weight.shape)
            markov_w2_shape = tuple(markov_head.markov_w2.weight.shape)
            logger.info(
                "Initialized DSpark draft runner. model=%s, block_size=%s, "
                "num_dspark_layers=%s, noise_token_id=%s, markov_rank=%s, "
                "confidence_threshold=%s, use_confidence_gate=%s, "
                "target_layer_ids=%s, decoder_layer_ids=%s, "
                "draft_config_compress_ratios=%s, "
                "draft_runtime_compress_ratios=%s, "
                "markov_w1_shape=%s, markov_w2_shape=%s",
                self.draft_model.__class__.__name__,
                self.block_size,
                self.num_dspark_layers,
                self.noise_token_id,
                self.markov_rank,
                self.confidence_threshold,
                self._use_confidence_gate,
                target_layer_ids,
                decoder_layer_ids,
                draft_config_compress_ratios,
                draft_runtime_compress_ratios,
                markov_w1_shape,
                markov_w2_shape,
            )

    def _is_tp0(self) -> bool:
        return self.tp_rank == 0

    def _init_fp8_wqkv_stack(self) -> None:
        if not envs.SGLANG_DSPARK_FP8_WQKV_STACK.get():
            return

        layers = getattr(self._draft_inner, "layers", None)
        if not layers:
            return

        weights = []
        scales = []
        biases = []
        kv_offsets = []
        out_sizes = []
        first_proj = None
        scale_name = None
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                self._log_fp8_wqkv_stack_disabled("missing self_attn")
                return

            if getattr(attn, "fuse_wqa_wkv", False):
                proj = getattr(attn, "wqkv_a", None)
                kv_start = int(attn.q_lora_rank)
                kv_end = kv_start + int(attn.head_dim)
            else:
                proj = getattr(attn, "wkv", None)
                kv_start = 0
                kv_end = int(attn.head_dim)

            cur_scale_name, scale = self._get_fp8_wqkv_scale(proj)
            if (
                proj is None
                or not hasattr(proj, "weight")
                or proj.weight.dtype
                not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                or scale is None
                or proj.weight.dim() != 2
                or scale.dim() != 2
                or kv_end > proj.weight.shape[0]
                or getattr(proj, "skip_bias_add", False)
            ):
                reason = (
                    "unsupported wqkv FP8 layout: "
                    f"weight_dtype={getattr(getattr(proj, 'weight', None), 'dtype', None)}, "
                    f"weight_shape={tuple(proj.weight.shape) if hasattr(proj, 'weight') else None}, "
                    f"scale_dtype={getattr(scale, 'dtype', None)}, "
                    f"scale_shape={tuple(scale.shape) if scale is not None else None}"
                )
                self._log_fp8_wqkv_stack_disabled(reason)
                return

            if first_proj is None:
                first_proj = proj
                scale_name = cur_scale_name
            elif (
                proj.weight.shape[1] != first_proj.weight.shape[1]
                or proj.weight.dtype != first_proj.weight.dtype
                or proj.quant_method.__class__ is not first_proj.quant_method.__class__
                or cur_scale_name != scale_name
                or scale.shape[1:] != scales[0].shape[1:]
            ):
                self._log_fp8_wqkv_stack_disabled("mixed wqkv FP8 layouts")
                return

            weights.append(proj.weight.detach())
            scales.append(scale.detach())
            kv_offsets.append((kv_start, kv_end))
            out_sizes.append(int(proj.weight.shape[0]))

            bias = getattr(proj, "bias", None)
            if bias is not None:
                biases.append(bias.detach())
            else:
                biases.append(None)

        if not weights:
            return

        has_bias = [bias is not None for bias in biases]
        if any(has_bias) and not all(has_bias):
            self._log_fp8_wqkv_stack_disabled("mixed wqkv bias layout")
            return

        stacked_weight = torch.cat(weights, dim=0).contiguous()
        stacked_scale = torch.cat(scales, dim=0).contiguous()
        stacked_bias = torch.cat(biases, dim=0).contiguous() if all(has_bias) else None

        stacked_proj = SimpleNamespace()
        for name in (
            "input_size",
            "input_size_per_partition",
            "params_dtype",
            "quant_config",
            "quant_method",
            "skip_bias_add",
            "weight_block_size",
        ):
            if hasattr(first_proj, name):
                setattr(stacked_proj, name, getattr(first_proj, name))
        stacked_proj.output_size = int(stacked_weight.shape[0])
        stacked_proj.output_size_per_partition = int(stacked_weight.shape[0])
        stacked_proj.weight = stacked_weight
        stacked_proj.bias = stacked_bias
        setattr(stacked_proj, scale_name, stacked_scale)

        self._stacked_wqkv_fp8_proj = stacked_proj
        self._stacked_wqkv_kv_offsets = kv_offsets
        self._stacked_wqkv_out_sizes = out_sizes

        if self._is_tp0():
            logger.info(
                "Enabled DSpark FP8 wqkv stack. layers=%s, weight_shape=%s, "
                "scale_shape=%s, scale_name=%s, kv_offsets=%s, "
                "env=SGLANG_DSPARK_FP8_WQKV_STACK",
                len(weights),
                tuple(stacked_weight.shape),
                tuple(stacked_scale.shape),
                scale_name,
                self._stacked_wqkv_kv_offsets,
            )

    def _log_fp8_wqkv_stack_disabled(self, reason: str) -> None:
        if self._is_tp0():
            logger.warning("DSpark FP8 wqkv stack disabled: %s", reason)

    @staticmethod
    def _get_fp8_wqkv_scale(proj) -> tuple[Optional[str], Optional[torch.Tensor]]:
        if proj is None:
            return None, None
        for name in ("weight_scale_inv", "weight_scale", "scale"):
            scale = getattr(proj, name, None)
            if scale is None:
                continue
            dtype = getattr(scale, "dtype", None)
            if dtype in (
                torch.uint8,
                torch.float32,
                torch.bfloat16,
                torch.float16,
                getattr(torch, "float8_e8m0fnu", None),
            ):
                return name, scale
        return None, None

    def _get_dp_decode_global_num_tokens(
        self, batch: ScheduleBatch
    ) -> Optional[list[int]]:
        if not self.server_args.enable_dp_attention or batch.global_num_tokens is None:
            return None

        return [int(x) for x in batch.global_num_tokens]

    def _get_target_aux_hidden_size(self) -> int:
        target_layer_ids = getattr(self._draft_inner, "target_layer_ids", None) or []
        hidden_size = int(getattr(self.model_runner.model_config, "hidden_size", 0))
        return hidden_size * max(1, len(target_layer_ids))

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
            self.draft_extend_attn_backend or self.draft_model_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker.init_attention_backends()
            self.draft_attn_backend = self.draft_model_runner.attn_backend
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.draft_model_runner,
                topk=1,
                speculative_num_steps=1,
            )
            self.draft_extend_attn_backend = (
                draft_backend_factory.create_draft_extend_backend()
            )

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            from sglang.srt.layers.attention import deepseek_v4_backend as _dsv4_be

            # Draft-block replay must use the same full-block attention metadata
            # as eager draft-block forward. Otherwise the captured graph can keep
            # causal block metadata even though runtime sets the DSpark flag.
            _dsv4_be._DSPARK_BLOCK_FULL_ATTN = int(self.block_size)
            try:
                self._draft_worker.init_cuda_graphs()
            finally:
                _dsv4_be._DSPARK_BLOCK_FULL_ATTN = 0

    def clear_cache_pool(self):
        pass

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _materialize_main_hidden_to_draft_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        projected: bool = False,
    ) -> None:
        if main_hidden is None:
            raise RuntimeError("DSpark missing target main_hidden context features.")
        if main_hidden.numel() == 0:
            return

        device = self.device
        if main_hidden.device != device:
            main_hidden = main_hidden.to(device, non_blocking=True)
        if cache_loc.device != device:
            cache_loc = cache_loc.to(device, non_blocking=True)
        if positions.device != device:
            positions = positions.to(device, non_blocking=True)
        if cache_loc.dtype != torch.int64:
            cache_loc = cache_loc.to(torch.int64)
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)

        attn_backend = self.draft_model_runner.attn_backend
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            torch.inference_mode(),
        ):
            # DeepSpec DSpark keeps target_hidden_states fixed for every draft
            # layer and computes ctx KV directly from hidden_norm(fc(target)).
            # The noise/block branch owns hc_pre/input_layernorm; applying it to
            # ctx hidden would change k_ctx/v_ctx. SGLang realizes cat(ctx, block)
            # through KV cache: write ctx KV here, then let draft_block write/read
            # block KV with full-block metadata.
            ctx_x = (
                main_hidden
                if projected
                else self.draft_model.project_main_hidden(main_hidden)
            )
            if self._stacked_wqkv_fp8_proj is None:
                self._last_context_kv_path_debug = "per_layer_kv_from_hidden"
                for layer in self._draft_inner.layers:
                    layer.self_attn.kv_from_hidden(
                        ctx_x, positions, cache_loc, attn_backend
                    )
            else:
                self._last_context_kv_path_debug = "stacked_fp8_wqkv"
                stacked_out = self._stacked_wqkv_fp8_proj.quant_method.apply(
                    self._stacked_wqkv_fp8_proj,
                    ctx_x,
                    self._stacked_wqkv_fp8_proj.bias,
                )
                layer_outputs = torch.split(
                    stacked_out, self._stacked_wqkv_out_sizes, dim=-1
                )
                for layer_idx, layer in enumerate(self._draft_inner.layers):
                    kv_start, kv_end = self._stacked_wqkv_kv_offsets[layer_idx]
                    self._write_draft_kv_from_projected_kv(
                        attn=layer.self_attn,
                        kv=layer_outputs[layer_idx][..., kv_start:kv_end],
                        positions=positions,
                        cache_loc=cache_loc,
                        attn_backend=attn_backend,
                    )

    def _materialize_main_hidden_to_draft_compressors(
        self,
        *,
        main_hidden: torch.Tensor,
        draft_forward_batch,
        projected: bool = False,
    ) -> None:
        if main_hidden is None:
            raise RuntimeError("DSpark missing target main_hidden context features.")
        if main_hidden.numel() == 0 or draft_forward_batch is None:
            return

        device = self.device
        if main_hidden.device != device:
            main_hidden = main_hidden.to(device, non_blocking=True)

        attn_backend = self.draft_model_runner.attn_backend
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            torch.inference_mode(),
        ):
            old_metadata = getattr(attn_backend, "forward_metadata", None)
            try:
                # The backend stores compressor metadata globally. Rebuild it for
                # this draft verify batch before replaying compressor writes with
                # target hidden; PD overlap can otherwise observe stale metadata.
                attn_backend.init_forward_metadata(draft_forward_batch)
                main_x = (
                    main_hidden
                    if projected
                    else self.draft_model.project_main_hidden(main_hidden)
                )
                for layer in self._draft_inner.layers:
                    attn = layer.self_attn
                    compressor = getattr(attn, "compressor", None)
                    if compressor is None:
                        continue
                    attn_backend.forward_core_compressor(
                        main_x,
                        draft_forward_batch,
                        attn.layer_id,
                        compressor,
                    )
            finally:
                attn_backend.forward_metadata = old_metadata

    def _materialize_main_hidden_to_draft_state(
        self,
        *,
        main_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        draft_forward_batch: Optional[ForwardBatch],
        kv_main_hidden: Optional[torch.Tensor] = None,
        projected: bool = False,
    ) -> None:
        try:
            self._materialize_main_hidden_to_draft_compressors(
                main_hidden=main_hidden,
                draft_forward_batch=draft_forward_batch,
                projected=projected,
            )
        except Exception as e:
            if self._is_tp0():
                logger.warning(
                    "DSpark draft compressor materialization failed; "
                    "skip compressor materialization for this call: %s",
                    e,
                )
        if kv_main_hidden is None:
            kv_main_hidden = main_hidden
        self._materialize_main_hidden_to_draft_kv(
            main_hidden=kv_main_hidden,
            cache_loc=cache_loc,
            positions=positions,
            projected=projected,
        )

    def _make_draft_prefill_forward_batch_for_materialize(
        self, batch: ScheduleBatch
    ) -> ForwardBatch:
        old_capture_hidden_mode = batch.capture_hidden_mode
        try:
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            draft_forward_batch = ForwardBatch.init_new(
                batch, self.draft_model_runner
            )
            draft_forward_batch.return_logprob = False
            draft_forward_batch.lora_ids = [None] * draft_forward_batch.batch_size
            return draft_forward_batch
        finally:
            batch.capture_hidden_mode = old_capture_hidden_mode

    def _make_draft_decode_forward_batch_for_materialize(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_before: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> ForwardBatch:
        old_input_ids = batch.input_ids
        old_out_cache_loc = batch.out_cache_loc
        old_spec_info = batch.spec_info
        old_forward_mode = batch.forward_mode
        old_capture_hidden_mode = batch.capture_hidden_mode
        old_seq_lens = batch.seq_lens
        old_seq_lens_cpu = batch.seq_lens_cpu
        old_seq_lens_sum = batch.seq_lens_sum
        try:
            batch.input_ids = torch.zeros(
                (seq_lens_before.numel(),), dtype=torch.int64, device=self.device
            )
            batch.out_cache_loc = cache_loc
            batch.spec_info = None
            batch.forward_mode = ForwardMode.DECODE
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            batch.seq_lens = seq_lens_before.to(dtype=old_seq_lens.dtype)
            batch.seq_lens_cpu = batch.seq_lens.detach().cpu()
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
            draft_forward_batch = ForwardBatch.init_new(
                batch, self.draft_model_runner
            )
            draft_forward_batch.return_logprob = False
            draft_forward_batch.lora_ids = [None] * draft_forward_batch.batch_size
            return draft_forward_batch
        finally:
            batch.input_ids = old_input_ids
            batch.out_cache_loc = old_out_cache_loc
            batch.spec_info = old_spec_info
            batch.forward_mode = old_forward_mode
            batch.capture_hidden_mode = old_capture_hidden_mode
            batch.seq_lens = old_seq_lens
            batch.seq_lens_cpu = old_seq_lens_cpu
            batch.seq_lens_sum = old_seq_lens_sum

    def _run_draft_bootstrap_forward(
        self,
        *,
        batch: ScheduleBatch,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        req_pool_indices: Optional[torch.Tensor] = None,
        dp_decode_global_num_tokens: Optional[list[int]] = None,
        num_tokens_per_req: int = 1,
        use_draft_extend_v2: bool = False,
    ) -> None:
        raise RuntimeError(
            "DSpark draft bootstrap via DRAFT_EXTEND_V2 is not DeepSpec-aligned. "
            "Use target-hidden context KV materialization plus normal draft_block."
        )

    @staticmethod
    def _prefill_bootstrap_global_req_counts(
        global_num_tokens: Optional[list[int]],
        tokens_per_req: int,
    ) -> Optional[list[int]]:
        if global_num_tokens is None or tokens_per_req <= 0:
            return None
        counts = []
        for num_tokens in global_num_tokens:
            num_tokens = int(num_tokens)
            if num_tokens <= 0:
                counts.append(0)
            else:
                counts.append(max(1, num_tokens // int(tokens_per_req)))
        return counts

    @staticmethod
    def _prefill_bootstrap_tokens_per_req(
        global_num_tokens: Optional[list[int]],
    ) -> int:
        if global_num_tokens is None:
            return 0
        return max((int(x) for x in global_num_tokens), default=0)

    @staticmethod
    def _prefill_bootstrap_predict_ids(
        batch: ScheduleBatch,
        next_token_ids: torch.Tensor,
        extend_lens: list[int],
    ) -> torch.Tensor:
        predict = batch.input_ids.clone()
        tail_tokens = next_token_ids.to(dtype=predict.dtype, device=predict.device)
        next_prompt_token = getattr(batch, "chunked_req_next_prompt_token", None)
        if next_prompt_token is not None:
            for i, req in enumerate(getattr(batch, "reqs", []) or []):
                if req is getattr(batch, "chunked_req", None):
                    tail_tokens = tail_tokens.clone()
                    tail_tokens[i] = next_prompt_token
                    break

        offset = 0
        for i, extend_len in enumerate(extend_lens):
            next_offset = offset + int(extend_len)
            if extend_len > 0:
                predict[offset:next_offset] = torch.cat(
                    (
                        predict[offset + 1 : next_offset],
                        tail_tokens[i].reshape(1),
                    )
                )
            offset = next_offset
        return predict

    def _materialize_disagg_prefill_hidden_to_draft_state(
        self,
        *,
        draft_input: DSparkDraftInputV2,
        batch: ScheduleBatch,
        prefix_lens: torch.Tensor,
    ) -> None:
        hidden = draft_input.hidden_states
        bs = len(prefix_lens)
        if hidden.numel() == 0 or bs == 0:
            return
        if hidden.dim() != 2:
            return
        if hidden.shape[0] != bs:
            if not self._is_tp0():
                return
            logger.warning(
                "Skip DSpark PD hidden bootstrap due to shape mismatch: "
                "hidden_shape=%s bs=%s",
                tuple(hidden.shape),
                bs,
            )
            return
        expected_hidden_size = self._get_target_aux_hidden_size()
        if expected_hidden_size and hidden.shape[-1] != expected_hidden_size:
            if not self._is_tp0():
                return
            logger.warning(
                "Skip DSpark PD hidden bootstrap due to hidden size mismatch: "
                "hidden_size=%s expected=%s",
                hidden.shape[-1],
                expected_hidden_size,
            )
            return
        hidden_valid_mask = draft_input.hidden_valid_mask
        if hidden_valid_mask is None or hidden_valid_mask.numel() != bs:
            return

        cache_loc = get_last_loc(
            self.model_runner.req_to_token_pool.req_to_token,
            batch.req_pool_indices,
            prefix_lens,
        )
        positions = prefix_lens.to(torch.int64) - 1
        valid = (prefix_lens > 0) & (cache_loc >= 0)
        valid &= hidden_valid_mask.to(
            device=valid.device, dtype=torch.bool, non_blocking=True
        )
        if not valid.any():
            return

        seq_lens_before = (prefix_lens[valid].to(torch.int64) - 1).clamp_min(0)
        draft_forward_batch = self._make_draft_decode_forward_batch_for_materialize(
            batch=batch,
            seq_lens_before=seq_lens_before,
            cache_loc=cache_loc[valid],
        )
        self._materialize_main_hidden_to_draft_state(
            main_hidden=hidden[valid],
            cache_loc=cache_loc[valid],
            positions=positions[valid],
            draft_forward_batch=draft_forward_batch,
        )

    def _pack_prefill_tail_hidden(
        self,
        *,
        hidden: torch.Tensor,
        extend_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = len(extend_lens)
        if hidden.numel() == 0 or bs == 0:
            return (
                torch.empty((0, 0), dtype=hidden.dtype, device=hidden.device),
                torch.empty((0, 0), dtype=torch.bool, device=hidden.device),
            )
        max_extend_len = max((int(x) for x in extend_lens), default=0)
        tail_len = min(128, max_extend_len)
        if tail_len <= 0:
            return (
                torch.empty((0, 0), dtype=hidden.dtype, device=hidden.device),
                torch.empty((0, 0), dtype=torch.bool, device=hidden.device),
            )
        tail_hidden = hidden.new_zeros((bs, tail_len, hidden.shape[-1]))
        tail_mask = torch.zeros((bs, tail_len), dtype=torch.bool, device=hidden.device)
        offset = 0
        for i, extend_len in enumerate(extend_lens):
            extend_len = int(extend_len)
            next_offset = offset + extend_len
            copy_len = min(tail_len, extend_len)
            if copy_len > 0:
                tail_hidden[i, tail_len - copy_len : tail_len].copy_(
                    hidden[next_offset - copy_len : next_offset]
                )
                tail_mask[i, tail_len - copy_len : tail_len] = True
            offset = next_offset
        return tail_hidden, tail_mask

    def _materialize_prefill_tail_hidden_to_draft_state(
        self,
        *,
        draft_input: DSparkDraftInputV2,
        batch: ScheduleBatch,
        prefix_lens: torch.Tensor,
    ) -> None:
        hidden = draft_input.prefill_tail_hidden_states
        mask = draft_input.prefill_tail_valid_mask
        bs = len(prefix_lens)
        if (
            hidden.numel() == 0
            or hidden.dim() != 3
            or mask is None
            or mask.shape != hidden.shape[:2]
            or hidden.shape[0] != bs
        ):
            return

        tail_len = hidden.shape[1]
        device = self.device
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        positions = (
            prefix_lens.to(device=device, dtype=torch.int64).unsqueeze(1)
            - tail_len
            + torch.arange(tail_len, device=device, dtype=torch.int64).unsqueeze(0)
        )
        valid = mask.to(device=device, dtype=torch.bool, non_blocking=True)
        valid &= positions >= 0
        if not valid.any():
            return

        req_pool_indices = batch.req_pool_indices.to(device=device, dtype=torch.int64)
        cache_locs = req_to_token[req_pool_indices.unsqueeze(1), positions.clamp_min(0)]
        valid &= cache_locs >= 0
        if not valid.any():
            return

        flat_hidden = hidden.to(device=device, non_blocking=True)[valid]
        flat_cache_locs = cache_locs[valid].to(torch.int64)
        flat_positions = positions[valid].to(torch.int64)
        if self._accept_anomaly_enabled:
            for i in range(bs):
                row_valid = valid[i]
                if not row_valid.any():
                    continue
                take = torch.nonzero(row_valid, as_tuple=False).view(-1)[-8:]
                self._record_boundary_debug(
                    req_pool_idx=int(req_pool_indices[i]),
                    stage="decode_tail_replay",
                    positions=positions[i, take],
                    cache_locs=cache_locs[i, take],
                    hidden_rows=hidden[i, take],
                    extra={"tail_len": int(tail_len)},
                )
        self._materialize_main_hidden_to_draft_state(
            main_hidden=flat_hidden,
            cache_loc=flat_cache_locs,
            positions=flat_positions,
            draft_forward_batch=None,
            projected=True,
        )

    def _write_draft_kv_from_projected_kv(
        self,
        *,
        attn,
        kv: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        attn_backend,
    ) -> None:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(cache_loc).to(
            torch.int32
        )
        token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=attn.layer_id,
            swa_loc=swa_loc,
            kv=kv,
            kv_weight=attn.kv_norm.weight.data,
            eps=attn.eps,
            freqs_cis=attn.freqs_cis,
            positions=positions,
        )

    def _run_draft_block(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        seq_lens_for_metadata: torch.Tensor,
        dp_decode_global_num_tokens: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, object]:
        draft_block_spec_info = DSparkDraftBlockInput(
            draft_token=block_ids.reshape(-1),
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        old_seq_lens = batch.seq_lens
        old_seq_lens_cpu = batch.seq_lens_cpu
        old_seq_lens_sum = batch.seq_lens_sum
        try:
            batch.seq_lens = seq_lens_for_metadata.to(dtype=old_seq_lens.dtype)
            batch.seq_lens_cpu = batch.seq_lens.detach().cpu()
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
            draft_forward_batch = draft_block_spec_info.prepare_for_draft_block(
                batch=batch,
                draft_model_runner=self.draft_model_runner,
                out_cache_loc=verify_out_cache_loc,
                dp_decode_global_num_tokens=dp_decode_global_num_tokens,
            )
        finally:
            batch.seq_lens = old_seq_lens
            batch.seq_lens_cpu = old_seq_lens_cpu
            batch.seq_lens_sum = old_seq_lens_sum

        from sglang.srt.layers.attention import deepseek_v4_backend as _dsv4_be

        _dsv4_be._DSPARK_BLOCK_FULL_ATTN = int(self.block_size)
        self._last_draft_block_ids_debug = [
            int(x)
            for x in block_ids.reshape(-1)[: min(int(block_ids.numel()), 8)]
            .detach()
            .cpu()
            .tolist()
        ]
        self._last_draft_visible_block_swa_locs_debug = None
        try:
            with torch.inference_mode():
                if os.getenv("SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH", "0") == "1":
                    graph_runner = self.draft_model_runner.decode_cuda_graph_runner
                    self.draft_model_runner.decode_cuda_graph_runner = None
                    try:
                        draft_runner_out = self.draft_model_runner.forward(
                            draft_forward_batch
                        )
                    finally:
                        self.draft_model_runner.decode_cuda_graph_runner = graph_runner
                else:
                    draft_runner_out = self.draft_model_runner.forward(
                        draft_forward_batch
                    )
                self._last_draft_block_metadata_debug = (
                    self._snapshot_draft_block_metadata(
                        draft_forward_batch=draft_forward_batch,
                        can_run_graph=bool(
                            getattr(draft_runner_out, "can_run_graph", False)
                        ),
                    )
                )
        finally:
            _dsv4_be._DSPARK_BLOCK_FULL_ATTN = 0

        raw = draft_runner_out.logits_output
        block_hidden = raw if isinstance(raw, torch.Tensor) else raw.hidden_states
        if block_hidden is None:
            raise RuntimeError("DSpark draft model returned no block hidden states.")
        reshape_bs = bs
        keep_tokens = bs * int(self.block_size)
        if bs == 0 and dp_decode_global_num_tokens is not None:
            reshape_bs = 1
            keep_tokens = int(self.block_size)
            if block_hidden.numel() == 0:
                block_hidden = block_hidden.new_zeros(
                    int(self.block_size), self._draft_inner.hidden_size
                )
        block_hidden = block_hidden[:keep_tokens]
        return (
            block_hidden.reshape(
                reshape_bs, int(self.block_size), block_hidden.shape[-1]
            ),
            draft_forward_batch,
        )

    def _snapshot_draft_block_metadata(
        self,
        *,
        draft_forward_batch,
        can_run_graph: bool,
    ) -> Optional[dict]:
        if not self._accept_anomaly_enabled:
            return None
        attn_backend = (
            getattr(self.draft_model_runner, "_dspark_last_graph_attn_backend", None)
            or self.draft_model_runner.attn_backend
        )
        metadata = getattr(attn_backend, "forward_metadata", None)
        if getattr(metadata, "core_attn_metadata", None) is None:
            inner_backends = getattr(attn_backend, "attn_backends", None)
            if inner_backends:
                metadata = getattr(inner_backends[0], "forward_metadata", None)
        core = getattr(metadata, "core_attn_metadata", None)
        if core is None:
            return {"draft_can_run_graph": can_run_graph, "draft_metadata": None}

        def first_last_rows(tensor):
            if (
                tensor is None
                or not isinstance(tensor, torch.Tensor)
                or tensor.numel() == 0
            ):
                return None
            rows = tensor.detach().cpu()
            if rows.ndim == 1:
                return [int(rows[0]), int(rows[-1])]
            return [
                [int(x) for x in rows[0].reshape(-1)[:8].tolist()],
                [int(x) for x in rows[-1].reshape(-1)[:8].tolist()],
            ]

        def first_last_row_edges(tensor):
            if (
                tensor is None
                or not isinstance(tensor, torch.Tensor)
                or tensor.numel() == 0
            ):
                return None
            rows = tensor.detach().cpu()
            if rows.ndim == 1:
                return None
            first = rows[0].reshape(-1)
            last = rows[-1].reshape(-1)
            return {
                "first_row_first8": [int(x) for x in first[:8].tolist()],
                "first_row_last8": [int(x) for x in first[-8:].tolist()],
                "last_row_first8": [int(x) for x in last[:8].tolist()],
                "last_row_last8": [int(x) for x in last[-8:].tolist()],
            }

        def valid_first_last_row_edges(indices, lengths):
            if (
                indices is None
                or lengths is None
                or not isinstance(indices, torch.Tensor)
                or not isinstance(lengths, torch.Tensor)
                or indices.numel() == 0
                or lengths.numel() == 0
            ):
                return None
            rows = indices.detach().cpu()
            lens = lengths.detach().cpu().reshape(-1)
            if rows.ndim == 1:
                return None
            out = {}
            for label, row_idx in (("first", 0), ("last", rows.shape[0] - 1)):
                row = rows[row_idx].reshape(-1)
                valid_len = int(lens[min(row_idx, lens.numel() - 1)])
                valid_len = max(0, min(valid_len, int(row.numel())))
                valid = row[:valid_len]
                out[f"{label}_valid_len"] = valid_len
                out[f"{label}_valid_first8"] = [
                    int(x) for x in valid[:8].tolist()
                ]
                out[f"{label}_valid_last8"] = [
                    int(x) for x in valid[-8:].tolist()
                ]
                block_locs = valid[-int(self.block_size) :]
                out[f"{label}_valid_block_locs"] = [
                    int(x) for x in block_locs.tolist()
                ]
            return out

        def valid_block_locs_by_row(indices, lengths):
            if (
                indices is None
                or lengths is None
                or not isinstance(indices, torch.Tensor)
                or not isinstance(lengths, torch.Tensor)
                or indices.numel() == 0
                or lengths.numel() == 0
            ):
                return None
            rows = indices.detach().cpu()
            lens = lengths.detach().cpu().reshape(-1)
            if rows.ndim == 1:
                return None
            max_rows = min(int(rows.shape[0]), int(self.block_size))
            out = []
            for row_idx in range(max_rows):
                row = rows[row_idx].reshape(-1)
                valid_len = int(lens[min(row_idx, lens.numel() - 1)])
                valid_len = max(0, min(valid_len, int(row.numel())))
                valid = row[:valid_len]
                block_locs = valid[-int(self.block_size) :]
                out.append([int(x) for x in block_locs.tolist()])
            return out

        swa_page_indices = getattr(core, "swa_page_indices", None)
        swa_topk_lengths = getattr(core, "swa_topk_lengths", None)
        valid_edges = valid_first_last_row_edges(swa_page_indices, swa_topk_lengths)
        valid_block_locs = valid_block_locs_by_row(swa_page_indices, swa_topk_lengths)
        self._last_draft_visible_block_swa_locs_debug = (
            None
            if valid_edges is None
            else valid_edges.get("last_valid_block_locs")
        )
        return {
            "draft_can_run_graph": can_run_graph,
            "draft_forward_mode": str(
                getattr(draft_forward_batch, "forward_mode", None)
            ),
            "draft_runtime_block_size": int(self.block_size),
            "target_verify_stride": int(self.verify_stride),
            "draft_spec_token_num": int(
                getattr(
                    getattr(draft_forward_batch, "spec_info", None),
                    "draft_token_num",
                    -1,
                )
            ),
            "draft_block_ids_first": self._last_draft_block_ids_debug,
            "draft_context_kv_path": getattr(
                self, "_last_context_kv_path_debug", None
            ),
            "draft_unified_kv": bool(
                getattr(
                    getattr(
                        self.draft_model_runner.attn_backend,
                        "token_to_kv_pool",
                        None,
                    ),
                    "_unified_kv",
                    False,
                )
            ),
            "draft_seq_lens_casual_first_last": first_last_rows(
                getattr(core, "seq_lens_casual", None)
            ),
            "draft_positions_casual_first_last": first_last_rows(
                getattr(core, "positions_casual", None)
            ),
            "draft_swa_topk_first_last": first_last_rows(
                swa_topk_lengths
            ),
            "draft_swa_indices_first_last": first_last_rows(
                swa_page_indices
            ),
            "draft_swa_indices_edges": first_last_row_edges(
                swa_page_indices
            ),
            "draft_swa_valid_edges": valid_edges,
            "draft_swa_valid_block_locs_by_row": valid_block_locs,
        }

    def _summarize_hidden_rows(self, rows: torch.Tensor) -> Optional[dict]:
        if rows is None or rows.numel() == 0:
            return None
        rows_f = rows.detach().float()
        row_norm = torch.linalg.vector_norm(rows_f, dim=-1)
        return {
            "shape": [int(x) for x in rows.shape],
            "sum": float(rows_f.sum().detach().cpu()),
            "abs_sum": float(rows_f.abs().sum().detach().cpu()),
            "max_abs": float(rows_f.abs().max().detach().cpu()),
            "row_norm_first_last": [
                float(row_norm[0].detach().cpu()),
                float(row_norm[-1].detach().cpu()),
            ],
            "row_sum_first_last": [
                float(rows_f[0].sum().detach().cpu()),
                float(rows_f[-1].sum().detach().cpu()),
            ],
        }

    @staticmethod
    def _summarize_weight_vector(weight: Optional[torch.Tensor]) -> Optional[dict]:
        if weight is None or weight.numel() == 0:
            return None
        weight_f = weight.detach().float()
        return {
            "shape": [int(x) for x in weight.shape],
            "l2": float(torch.linalg.vector_norm(weight_f).detach().cpu()),
            "abs_mean": float(weight_f.abs().mean().detach().cpu()),
            "min": float(weight_f.min().detach().cpu()),
            "max": float(weight_f.max().detach().cpu()),
        }

    def _target_aux_payload(
        self,
        *,
        raw_hidden_rows: torch.Tensor,
        projected_rows: torch.Tensor,
    ) -> Optional[dict]:
        if (
            raw_hidden_rows is None
            or projected_rows is None
            or raw_hidden_rows.numel() == 0
            or projected_rows.numel() == 0
        ):
            return None
        target_layer_ids = list(getattr(self._draft_inner, "target_layer_ids", []) or [])
        hidden_size = int(getattr(self._draft_inner, "hidden_size", 0))
        raw = raw_hidden_rows.detach()
        projected = projected_rows.detach()
        raw_norm = torch.linalg.vector_norm(raw.float(), dim=-1)
        projected_norm = torch.linalg.vector_norm(projected.float(), dim=-1)
        norm_ratio = projected_norm / raw_norm.clamp_min(1e-6)
        decoder_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
        payload = {
            "target_layer_ids": [int(x) for x in target_layer_ids],
            "decoder_layer_ids": decoder_layer_ids,
            "raw": self._summarize_hidden_rows(raw),
            "projected": self._summarize_hidden_rows(projected),
            "projected_to_raw_norm_ratio_first_last": [
                float(norm_ratio[0].detach().cpu()),
                float(norm_ratio[-1].detach().cpu()),
            ],
            "main_norm_weight": self._summarize_weight_vector(
                getattr(getattr(self._draft_inner, "main_norm", None), "weight", None)
            ),
            "shared_norm_weight": self._summarize_weight_vector(
                getattr(
                    getattr(getattr(self._draft_inner, "shared_head", None), "norm", None),
                    "weight",
                    None,
                )
            ),
        }
        if target_layer_ids and hidden_size > 0 and raw.shape[-1] == (
            len(target_layer_ids) * hidden_size
        ):
            raw_layers = raw.reshape(raw.shape[0], len(target_layer_ids), hidden_size)
            payload["raw_layers"] = [
                {
                    "target_layer_id": int(layer_id),
                    "decoder_layer_id": int(decoder_layer_ids[i]),
                    "hidden": self._summarize_hidden_rows(raw_layers[:, i, :]),
                }
                for i, layer_id in enumerate(target_layer_ids)
            ]
        else:
            payload["expected_raw_width"] = int(len(target_layer_ids) * hidden_size)
        return payload

    def _record_target_aux_debug(
        self,
        *,
        req_pool_idx: int,
        stage: str,
        raw_hidden_rows: torch.Tensor,
        projected_rows: torch.Tensor,
    ) -> None:
        if not self._accept_anomaly_enabled or not self._is_tp0():
            return
        try:
            payload = self._target_aux_payload(
                raw_hidden_rows=raw_hidden_rows.to(device=self.device),
                projected_rows=projected_rows.to(device=self.device),
            )
            if payload is None:
                return
            debug = self._target_aux_debug_by_req_pool.setdefault(
                int(req_pool_idx), {}
            )
            debug[stage] = payload
        except Exception as e:
            debug = self._target_aux_debug_by_req_pool.setdefault(
                int(req_pool_idx), {}
            )
            debug[stage] = {"error": str(e)}

    def _boundary_loc_payload(
        self,
        *,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        hidden_rows: torch.Tensor,
    ) -> dict:
        token_to_kv_pool = self.draft_model_runner.attn_backend.token_to_kv_pool
        translate_swa = getattr(token_to_kv_pool, "translate_loc_from_full_to_swa", None)
        swa_locs = None
        if translate_swa is not None and cache_locs.numel() > 0:
            swa_locs = translate_swa(cache_locs.to(device=self.device, dtype=torch.int64))
        return {
            "positions": [int(x) for x in positions.detach().cpu().tolist()],
            "full_locs": [int(x) for x in cache_locs.detach().cpu().tolist()],
            "swa_locs": (
                [int(x) for x in swa_locs.detach().cpu().tolist()]
                if swa_locs is not None
                else None
            ),
            "hidden": self._summarize_hidden_rows(hidden_rows),
        }

    def _record_boundary_debug(
        self,
        *,
        req_pool_idx: int,
        stage: str,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        hidden_rows: torch.Tensor,
        extra: Optional[dict] = None,
    ) -> None:
        if not self._accept_anomaly_enabled:
            return
        if not self._is_tp0():
            return
        if positions.numel() == 0 or cache_locs.numel() == 0 or hidden_rows.numel() == 0:
            return
        try:
            payload = self._boundary_loc_payload(
                positions=positions.to(device=self.device, dtype=torch.int64),
                cache_locs=cache_locs.to(device=self.device, dtype=torch.int64),
                hidden_rows=hidden_rows.to(device=self.device),
            )
            if extra:
                payload.update(extra)
            debug = self._boundary_debug_by_req_pool.setdefault(int(req_pool_idx), {})
            debug[stage] = payload
            if stage == "decode_verify_write":
                history = debug.setdefault("decode_verify_write_history", [])
                history.append(payload)
                if len(history) > 16:
                    del history[:-16]
            self._record_swa_write_source(
                req_pool_idx=int(req_pool_idx),
                stage=stage,
                payload=payload,
            )
        except Exception as e:
            debug = self._boundary_debug_by_req_pool.setdefault(int(req_pool_idx), {})
            debug[stage] = {"error": str(e)}

    def _record_swa_write_source(
        self,
        *,
        req_pool_idx: int,
        stage: str,
        payload: dict,
    ) -> None:
        swa_locs = payload.get("swa_locs")
        if not isinstance(swa_locs, list):
            return
        source = stage
        if stage == "decode_tail_replay":
            source = "prefill_tail_replay"
        elif stage == "prefill_tail_source":
            source = "prefill_source"
        elif stage == "decode_verify_write":
            source = "decode_verify_write"
        self._record_swa_locs_source(
            req_pool_idx=int(req_pool_idx), swa_locs=swa_locs, source=source
        )

    def _record_swa_locs_source(
        self,
        *,
        req_pool_idx: int,
        swa_locs: list[int],
        source: str,
    ) -> None:
        source_by_loc = self._swa_write_source_by_req_pool.setdefault(req_pool_idx, {})
        for loc in swa_locs:
            if loc is None:
                continue
            source_by_loc[int(loc)] = source
        if len(source_by_loc) > 512:
            # Keep only the tail of the sliding-window history.
            for loc in sorted(source_by_loc)[:-256]:
                source_by_loc.pop(loc, None)

    def _record_draft_block_write_sources(
        self, req_pool_indices: torch.Tensor, draft_out_cache_loc: torch.Tensor
    ) -> None:
        if not self._accept_anomaly_enabled or not self._is_tp0():
            return
        if req_pool_indices.numel() == 0 or draft_out_cache_loc.numel() == 0:
            return
        try:
            locs = draft_out_cache_loc.to(device=self.device, dtype=torch.int64)
            token_to_kv_pool = self.draft_model_runner.attn_backend.token_to_kv_pool
            translate_swa = getattr(token_to_kv_pool, "translate_loc_from_full_to_swa", None)
            if translate_swa is not None:
                locs = translate_swa(locs)
            locs_2d = locs.view(req_pool_indices.numel(), int(self.block_size))
            for i, req_pool_idx in enumerate(req_pool_indices.detach().cpu().tolist()):
                self._record_swa_locs_source(
                    req_pool_idx=int(req_pool_idx),
                    swa_locs=[int(x) for x in locs_2d[i].detach().cpu().tolist()],
                    source="draft_block_write",
                )
        except Exception:
            return

    def _visible_window_source_debug(
        self, req_pool_idx: int, visible_locs: Optional[list[int]]
    ) -> Optional[dict]:
        if not visible_locs:
            return None
        source_by_loc = self._swa_write_source_by_req_pool.get(int(req_pool_idx), {})
        sources = [source_by_loc.get(int(loc), "unknown") for loc in visible_locs]
        counts: dict[str, int] = {}
        for source in sources:
            counts[source] = counts.get(source, 0) + 1
        return {
            "visible_locs": [int(x) for x in visible_locs],
            "sources": sources,
            "counts": counts,
        }

    def _checksum_swa_kv_rows(
        self,
        *,
        layer_id: int,
        swa_locs: torch.Tensor,
    ) -> Optional[dict]:
        try:
            token_to_kv_pool = self.draft_model_runner.attn_backend.token_to_kv_pool
            get_swa_raw_buffer = getattr(token_to_kv_pool, "get_swa_raw_buffer", None)
            if get_swa_raw_buffer is None or swa_locs.numel() == 0:
                return None
            valid = swa_locs.to(device=self.device, dtype=torch.int64)
            valid = valid[valid >= 0]
            if valid.numel() == 0:
                return None
            valid = valid[: min(int(valid.numel()), 16)]
            raw = get_swa_raw_buffer(int(layer_id))
            swa_pool = getattr(token_to_kv_pool, "swa_kv_pool", None)
            page_size = int(getattr(swa_pool, "page_size", 1))
            page_ids = torch.div(valid, page_size, rounding_mode="floor")
            page_offsets = valid % page_size
            in_bounds = page_ids < int(raw.shape[0])
            page_ids = page_ids[in_bounds]
            page_offsets = page_offsets[in_bounds]
            valid = valid[in_bounds]
            if valid.numel() == 0:
                return None
            page_rows = raw[page_ids].reshape(valid.shape[0], page_size, -1)
            rows = page_rows[
                torch.arange(valid.shape[0], device=valid.device), page_offsets
            ]
            rows_f = rows.float()
            return {
                "layer_id": int(layer_id),
                "raw_shape": [int(x) for x in raw.shape],
                "page_size": page_size,
                "num_rows": int(valid.numel()),
                "swa_locs": [int(x) for x in valid.detach().cpu().tolist()],
                "page_ids": [int(x) for x in page_ids.detach().cpu().tolist()],
                "page_offsets": [
                    int(x) for x in page_offsets.detach().cpu().tolist()
                ],
                "sum": float(rows_f.sum().detach().cpu()),
                "abs_sum": float(rows_f.abs().sum().detach().cpu()),
                "max_abs": float(rows_f.abs().max().detach().cpu()),
            }
        except Exception as e:
            return {"layer_id": int(layer_id), "error": str(e)}

    def _build_accept_anomaly_kv_debug(
        self,
        *,
        req_pool_idx: int,
        prefix_len: int,
        block_size: int,
    ) -> Optional[dict]:
        try:
            req_to_token = self.model_runner.req_to_token_pool.req_to_token
            token_to_kv_pool = self.draft_model_runner.attn_backend.token_to_kv_pool
            translate_swa = getattr(token_to_kv_pool, "translate_loc_from_full_to_swa", None)
            if translate_swa is None:
                return None
            anchor_len = max(int(prefix_len) - 1, 0)
            start = max(anchor_len - 128, 0)
            end = anchor_len + int(block_size)
            if end <= start:
                return None
            logical = torch.arange(start, end, device=req_to_token.device, dtype=torch.int64)
            full_locs = req_to_token[
                torch.tensor(int(req_pool_idx), device=req_to_token.device).view(1),
                logical,
            ].view(-1)
            swa_locs = translate_swa(full_locs)
            verify_start = int(prefix_len)
            verify_end = verify_start + int(block_size)
            verify_logical = torch.arange(
                verify_start,
                verify_end,
                device=req_to_token.device,
                dtype=torch.int64,
            )
            verify_full_locs = req_to_token[
                torch.tensor(int(req_pool_idx), device=req_to_token.device).view(1),
                verify_logical,
            ].view(-1)
            verify_swa_locs = translate_swa(verify_full_locs)
            layers = getattr(self._draft_inner, "layers", [])
            layer_ids = []
            if layers:
                layer_ids.append(int(layers[0].self_attn.layer_id))
                if len(layers) > 1:
                    layer_ids.append(int(layers[-1].self_attn.layer_id))
            visible_block_swa_locs = self._last_draft_visible_block_swa_locs_debug
            visible_block_swa = (
                torch.tensor(
                    visible_block_swa_locs,
                    device=swa_locs.device,
                    dtype=torch.int64,
                )
                if visible_block_swa_locs
                else None
            )
            context_len = max(int(swa_locs.numel()) - int(block_size), 0)
            context_head_swa = swa_locs[: min(8, context_len)]
            context_tail_start = max(context_len - 8, 0)
            context_tail_swa = swa_locs[context_tail_start:context_len]
            draft_query_swa = swa_locs[-int(block_size) :]
            return {
                "prefix_len": int(prefix_len),
                "anchor_len": int(anchor_len),
                "logical_first_last": [int(start), int(end - 1)],
                "full_first8": [int(x) for x in full_locs[:8].detach().cpu().tolist()],
                "full_last8": [int(x) for x in full_locs[-8:].detach().cpu().tolist()],
                "swa_first8": [int(x) for x in swa_locs[:8].detach().cpu().tolist()],
                "swa_last8": [int(x) for x in swa_locs[-8:].detach().cpu().tolist()],
                "context_len": context_len,
                "context_tail_swa_locs": [
                    int(x) for x in context_tail_swa.detach().cpu().tolist()
                ],
                "draft_query_swa_locs": [
                    int(x) for x in draft_query_swa.detach().cpu().tolist()
                ],
                "verify_token_logical_locs": [
                    int(x) for x in verify_logical.detach().cpu().tolist()
                ],
                "verify_token_swa_locs": [
                    int(x) for x in verify_swa_locs.detach().cpu().tolist()
                ],
                "visible_block_swa_locs": visible_block_swa_locs,
                "visible_matches_draft_query": (
                    visible_block_swa_locs
                    == [int(x) for x in draft_query_swa.detach().cpu().tolist()]
                    if visible_block_swa_locs is not None
                    else None
                ),
                "boundary_debug": self._boundary_debug_by_req_pool.get(
                    int(req_pool_idx)
                ),
                "target_aux_debug": self._target_aux_debug_by_req_pool.get(
                    int(req_pool_idx)
                ),
                "context_head_kv_checksums": [
                    self._checksum_swa_kv_rows(
                        layer_id=layer_id,
                        swa_locs=context_head_swa,
                    )
                    for layer_id in layer_ids
                ],
                "context_tail_kv_checksums": [
                    self._checksum_swa_kv_rows(
                        layer_id=layer_id,
                        swa_locs=context_tail_swa,
                    )
                    for layer_id in layer_ids
                ],
                "draft_query_kv_checksums": [
                    self._checksum_swa_kv_rows(
                        layer_id=layer_id,
                        swa_locs=draft_query_swa,
                    )
                    for layer_id in layer_ids
                ],
                "visible_block_kv_checksums": (
                    [
                        self._checksum_swa_kv_rows(
                            layer_id=layer_id,
                            swa_locs=visible_block_swa,
                        )
                        for layer_id in layer_ids
                    ]
                    if visible_block_swa is not None
                    else None
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    def _ensure_markov_refine_buffers(self, bs: int, device: torch.device) -> None:
        cap = self._markov_refine_buffer_cap
        if (
            cap >= int(bs)
            and self._markov_candidates_buf is not None
            and self._markov_embeds_buf is not None
            and self._markov_candidates_buf.device == device
            and self._markov_embeds_buf.device == device
        ):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        markov_weight = getattr(self._draft_inner.markov_head.markov_w1, "weight", None)
        markov_dtype = (
            markov_weight.dtype
            if markov_weight is not None
            else self.draft_model.lm_head.weight.dtype
        )
        self._markov_candidates_buf = torch.empty(
            (new_cap, int(self.verify_stride)), dtype=torch.int64, device=device
        )
        self._markov_embeds_buf = torch.empty(
            (new_cap, int(self.block_size), int(self.markov_rank)),
            dtype=markov_dtype,
            device=device,
        )
        self._markov_refine_buffer_cap = new_cap

    def _refine_block_markov(
        self,
        *,
        block_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        output_bs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = int(block_hidden.shape[0])
        output_bs = bs if output_bs is None else int(output_bs)
        block_size = int(self.block_size)
        verify_stride = int(self.verify_stride)
        if bs == 0:
            empty_tokens = torch.empty(
                (output_bs, verify_stride),
                dtype=torch.int64,
                device=block_hidden.device,
            )
            empty_confidence = torch.empty(
                (output_bs, block_size), dtype=torch.float32, device=block_hidden.device
            )
            return empty_tokens, empty_confidence

        self._ensure_markov_refine_buffers(bs, block_hidden.device)
        assert self._markov_candidates_buf is not None
        assert self._markov_embeds_buf is not None
        candidates = self._markov_candidates_buf[:bs]
        markov_embeds = self._markov_embeds_buf[:bs]

        markov_head = self._draft_inner.markov_head
        confidence_head = self._draft_inner.confidence_head
        lm_head = self.draft_model.lm_head
        logits_processor = self.draft_model.logits_processor

        vocab_size = int(self._draft_inner.vocab_size)

        def _gather_full_vocab(logits_shard: torch.Tensor, head) -> torch.Tensor:
            def _reindex_sharded_vocab(logits: torch.Tensor) -> torch.Tensor:
                mapping_fn = getattr(head, "get_sharded_to_full_mapping", None)
                if mapping_fn is None:
                    return logits[..., :vocab_size]
                mapping = mapping_fn()
                if mapping is None:
                    return logits[..., :vocab_size]
                cache_key = (id(head), int(logits.shape[-1]), logits.device)
                mapping_tensor = self._vocab_shard_mapping_cache.get(cache_key)
                if mapping_tensor is None or mapping_tensor.numel() != len(mapping):
                    mapping_tensor = torch.tensor(
                        mapping, dtype=torch.long, device=logits.device
                    )
                    self._vocab_shard_mapping_cache[cache_key] = mapping_tensor
                return logits.index_select(-1, mapping_tensor)[..., :vocab_size]

            if logits_shard.shape[-1] >= vocab_size:
                return _reindex_sharded_vocab(logits_shard)
            if getattr(head, "use_attn_tp_group", False):
                group = get_attn_tp_group()
                if group.world_size == 1:
                    return logits_shard[..., :vocab_size]
                logits = group.all_gather(logits_shard, dim=-1)
                return _reindex_sharded_vocab(logits)
            tp_size = get_tensor_model_parallel_world_size()
            if tp_size == 1:
                return logits_shard[..., :vocab_size]
            logits = tensor_model_parallel_all_gather(logits_shard, dim=-1)
            return _reindex_sharded_vocab(logits)

        def _compute_full_vocab_logits(
            hidden_states: torch.Tensor,
            head,
        ) -> torch.Tensor:
            logits_shard = logits_processor._compute_lm_head(hidden_states, head)
            return _gather_full_vocab(logits_shard, head)

        if anchor_tokens.numel() == bs:
            first_tokens = anchor_tokens.view(-1).to(torch.int64)
        else:
            first_tokens = torch.full(
                (bs,), self.noise_token_id, dtype=torch.int64, device=block_hidden.device
            )
        candidates[:, 0].copy_(first_tokens)

        with torch.inference_mode():
            base_logits = _compute_full_vocab_logits(block_hidden, lm_head)
            debug_enabled = bool(self._accept_anomaly_enabled)
            if debug_enabled:
                debug_topk_k = (
                    min(5, int(base_logits.shape[-1]))
                    if self._accept_anomaly_topk_enabled
                    else 0
                )
                if debug_topk_k > 0:
                    base_topk_values, base_topk_ids = torch.topk(
                        base_logits, k=debug_topk_k, dim=-1
                    )
                else:
                    base_topk_ids = None
                    base_topk_values = None
                base_top1 = torch.argmax(base_logits, dim=-1)
                base_top1_logit = torch.gather(
                    base_logits, dim=-1, index=base_top1.unsqueeze(-1)
                ).squeeze(-1)
                hidden_f = block_hidden.float()
                hidden_norm = hidden_f.norm(dim=-1)
                hidden_abs_mean = hidden_f.abs().mean(dim=-1)
                if block_size > 1:
                    hidden_cos_adjacent = torch.nn.functional.cosine_similarity(
                        hidden_f[:, :-1, :], hidden_f[:, 1:, :], dim=-1
                    )
                else:
                    hidden_cos_adjacent = hidden_f.new_empty((bs, 0))
                markov_top1 = torch.empty(
                    (bs, block_size), dtype=torch.int64, device=block_hidden.device
                )
                if debug_topk_k > 0:
                    bias_topk_ids = torch.empty(
                        (bs, block_size, debug_topk_k),
                        dtype=torch.int64,
                        device=block_hidden.device,
                    )
                    bias_topk_values = torch.empty(
                        (bs, block_size, debug_topk_k),
                        dtype=base_logits.dtype,
                        device=block_hidden.device,
                    )
                    final_topk_ids = torch.empty(
                        (bs, block_size, debug_topk_k),
                        dtype=torch.int64,
                        device=block_hidden.device,
                    )
                    final_topk_values = torch.empty(
                        (bs, block_size, debug_topk_k),
                        dtype=base_logits.dtype,
                        device=block_hidden.device,
                    )
                else:
                    bias_topk_ids = None
                    bias_topk_values = None
                    final_topk_ids = None
                    final_topk_values = None
                prev_token_debug = torch.empty_like(markov_top1)
            else:
                debug_topk_k = 0
                base_topk_ids = None
                base_topk_values = None
                base_top1 = None
                base_top1_logit = None
                hidden_norm = None
                hidden_abs_mean = None
                hidden_cos_adjacent = None
                markov_top1 = None
                bias_topk_ids = None
                bias_topk_values = None
                final_topk_ids = None
                final_topk_values = None
                prev_token_debug = None
            prev_tokens = candidates[:, 0]
            for i in range(block_size):
                if debug_enabled:
                    assert prev_token_debug is not None
                    prev_token_debug[:, i].copy_(prev_tokens)
                prev_embed = markov_head.get_prev_embeddings(prev_tokens)
                markov_embeds[:, i].copy_(prev_embed)
                bias = _compute_full_vocab_logits(prev_embed, markov_head.markov_w2)
                if debug_enabled and debug_topk_k > 0:
                    assert bias_topk_ids is not None
                    assert bias_topk_values is not None
                    step_bias_topk_values, step_bias_topk_ids = torch.topk(
                        bias, k=debug_topk_k, dim=-1
                    )
                    bias_topk_ids[:, i].copy_(step_bias_topk_ids)
                    bias_topk_values[:, i].copy_(step_bias_topk_values)
                bias.add_(base_logits[:, i])
                next_tokens = torch.argmax(bias, dim=-1)
                if debug_enabled:
                    assert markov_top1 is not None
                    if debug_topk_k > 0:
                        assert final_topk_ids is not None
                        assert final_topk_values is not None
                        step_topk_values, step_topk_ids = torch.topk(
                            bias, k=debug_topk_k, dim=-1
                        )
                        final_topk_ids[:, i].copy_(step_topk_ids)
                        final_topk_values[:, i].copy_(step_topk_values)
                    markov_top1[:, i].copy_(next_tokens)
                candidates[:, i + 1].copy_(next_tokens)
                prev_tokens = next_tokens

            confidence = confidence_head(block_hidden, markov_embeds)
            if debug_enabled:
                assert base_top1 is not None
                assert base_top1_logit is not None
                assert hidden_norm is not None
                assert hidden_abs_mean is not None
                assert hidden_cos_adjacent is not None
                assert markov_top1 is not None
                assert prev_token_debug is not None
                self._last_markov_refine_debug = {
                    "base_topk_ids": (
                        base_topk_ids[:output_bs].detach()
                        if base_topk_ids is not None
                        else None
                    ),
                    "base_topk_values": (
                        base_topk_values[:output_bs].detach()
                        if base_topk_values is not None
                        else None
                    ),
                    "base_top1": base_top1[:output_bs].detach(),
                    "base_top1_logit": base_top1_logit[:output_bs].detach(),
                    "hidden_norm": hidden_norm[:output_bs].detach(),
                    "hidden_abs_mean": hidden_abs_mean[:output_bs].detach(),
                    "hidden_cos_adjacent": hidden_cos_adjacent[:output_bs].detach(),
                    "markov_top1": markov_top1[:output_bs].detach(),
                    "bias_topk_ids": (
                        bias_topk_ids[:output_bs].detach()
                        if bias_topk_ids is not None
                        else None
                    ),
                    "bias_topk_values": (
                        bias_topk_values[:output_bs].detach()
                        if bias_topk_values is not None
                        else None
                    ),
                    "final_topk_ids": (
                        final_topk_ids[:output_bs].detach()
                        if final_topk_ids is not None
                        else None
                    ),
                    "final_topk_values": (
                        final_topk_values[:output_bs].detach()
                        if final_topk_values is not None
                        else None
                    ),
                    "prev_tokens": prev_token_debug[:output_bs].detach(),
                }
            else:
                self._last_markov_refine_debug = None

        return candidates[:output_bs], confidence[:output_bs]

    def _confident_prefix(self, confidence: torch.Tensor) -> torch.Tensor:
        if not self._use_confidence_gate:
            return torch.full(
                (confidence.shape[0],),
                int(confidence.shape[1]),
                dtype=torch.int32,
                device=confidence.device,
            )
        keep = torch.sigmoid(confidence) >= self.confidence_threshold
        return keep.to(torch.int32).cumprod(dim=1).sum(dim=1)

    def _ensure_accept_bonus_buffers(self, bs: int) -> None:
        if self._accept_bonus_buffer_cap >= int(bs):
            return

        new_cap = max(
            int(bs),
            (
                self._accept_bonus_buffer_cap * 2
                if self._accept_bonus_buffer_cap > 0
                else int(bs)
            ),
        )
        device = self.device
        verify_stride = int(self.verify_stride)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, verify_stride), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._new_seq_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._accept_bonus_buffer_cap = new_cap

    def _next_accept_bonus_buffers(self, bs: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_accept_bonus_buffers(bs)
        slot = self._accept_bonus_buffer_slot
        self._accept_bonus_buffer_slot = (slot + 1) % 2
        return (
            self._commit_lens_bufs[slot][:bs],
            self._bonus_id_bufs[slot][:bs],
            self._out_tokens_bufs[slot][:bs],
            self._new_seq_lens_bufs[slot][:bs],
        )

    def _compute_accept_bonus_eager(
        self,
        *,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, verify_stride = candidates.shape
        if target_predict.shape != candidates.shape:
            raise ValueError(
                "target_predict must match DSpark candidates shape. "
                f"candidates.shape={tuple(candidates.shape)}, "
                f"target_predict.shape={tuple(target_predict.shape)}"
            )
        draft_width = verify_stride - 1
        if confidence.shape[0] != bs or confidence.shape[1] != draft_width:
            raise ValueError(
                "confidence must have one row per sampled DSpark draft token. "
                f"confidence.shape={tuple(confidence.shape)}, "
                f"candidates.shape={tuple(candidates.shape)}"
            )
        if draft_width > 0:
            matches = candidates[:, 1:] == target_predict[:, :draft_width]
            correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
        else:
            correct_len = torch.zeros((bs,), dtype=torch.int32, device=candidates.device)
        confident_prefix = self._confident_prefix(confidence)
        correct_len = torch.minimum(
            correct_len.to(torch.int64), confident_prefix.to(torch.int64)
        )
        bonus_tokens = target_predict.gather(1, correct_len.unsqueeze(1)).squeeze(1)
        commit_lens = correct_len.to(torch.int32) + 1

        out_tokens = torch.empty(
            (bs, verify_stride), dtype=torch.int64, device=candidates.device
        )
        if draft_width > 0:
            out_tokens[:, :draft_width].copy_(candidates[:, 1:])
        out_tokens[:, draft_width].fill_(0)
        out_tokens.scatter_(
            1,
            correct_len.unsqueeze(1),
            bonus_tokens.unsqueeze(1).to(torch.int64),
        )
        return commit_lens, bonus_tokens, out_tokens

    def _build_accept_anomaly_markov_debug(
        self,
        *,
        row_idx: int,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Optional[dict]:
        try:
            debug = self._last_markov_refine_debug
            if not debug:
                return None
            base_topk_ids = debug.get("base_topk_ids")
            base_topk_values = debug.get("base_topk_values")
            base_top1 = debug.get("base_top1")
            base_top1_logit = debug.get("base_top1_logit")
            hidden_norm = debug.get("hidden_norm")
            hidden_abs_mean = debug.get("hidden_abs_mean")
            hidden_cos_adjacent = debug.get("hidden_cos_adjacent")
            markov_top1 = debug.get("markov_top1")
            bias_topk_ids = debug.get("bias_topk_ids")
            bias_topk_values = debug.get("bias_topk_values")
            final_topk_ids = debug.get("final_topk_ids")
            final_topk_values = debug.get("final_topk_values")
            prev_tokens = debug.get("prev_tokens")
            if (
                base_top1 is None
                or base_top1_logit is None
                or hidden_norm is None
                or hidden_abs_mean is None
                or hidden_cos_adjacent is None
                or markov_top1 is None
                or prev_tokens is None
            ):
                return None
            if int(row_idx) >= int(markov_top1.shape[0]):
                return None
            verify_stride = int(candidates.shape[1])
            draft_width = int(markov_top1.shape[1])
            candidate_limit = min(verify_stride, 8)
            draft_limit = min(draft_width, 8)
            candidate_row = candidates[row_idx]
            target_row = target_predict[row_idx]
            confidence_row = confidence[row_idx]
            has_topk_debug = (
                base_topk_ids is not None
                and base_topk_values is not None
                and bias_topk_ids is not None
                and bias_topk_values is not None
                and final_topk_ids is not None
                and final_topk_values is not None
            )
            if has_topk_debug:
                row_base_topk_ids = (
                    base_topk_ids[row_idx, :draft_limit].detach().cpu()
                )
                row_base_topk_values = (
                    base_topk_values[row_idx, :draft_limit].detach().cpu()
                )
                row_bias_topk_ids = (
                    bias_topk_ids[row_idx, :draft_limit].detach().cpu()
                )
                row_bias_topk_values = (
                    bias_topk_values[row_idx, :draft_limit].detach().cpu()
                )
                row_final_topk_ids = (
                    final_topk_ids[row_idx, :draft_limit].detach().cpu()
                )
                row_final_topk_values = (
                    final_topk_values[row_idx, :draft_limit].detach().cpu()
                )
            else:
                row_base_topk_ids = None
                row_base_topk_values = None
                row_bias_topk_ids = None
                row_bias_topk_values = None
                row_final_topk_ids = None
                row_final_topk_values = None
            row_base = base_top1[row_idx, :draft_limit].detach().cpu()
            row_base_top1_logit = base_top1_logit[
                row_idx, :draft_limit
            ].detach().cpu()
            row_hidden_norm = hidden_norm[row_idx, :draft_limit].detach().cpu()
            row_hidden_abs_mean = hidden_abs_mean[
                row_idx, :draft_limit
            ].detach().cpu()
            norm_weight = self._draft_inner.shared_head.norm.weight.detach().float()
            norm_weight_l2 = float(norm_weight.norm().detach().cpu())
            norm_weight_abs_mean = float(norm_weight.abs().mean().detach().cpu())
            row_hidden_l2_ratio = row_hidden_norm / max(norm_weight_l2, 1e-6)
            row_hidden_cos_adjacent = (
                hidden_cos_adjacent[row_idx, : max(draft_limit - 1, 0)]
                .detach()
                .cpu()
            )
            row_markov = markov_top1[row_idx].detach().cpu()
            row_prev = prev_tokens[row_idx, :draft_limit].detach().cpu()
            candidate_target_hits = []
            for cand_idx in range(1, min(verify_stride, 7)):
                hits = torch.nonzero(
                    target_row[:candidate_limit] == candidate_row[cand_idx],
                    as_tuple=False,
                ).view(-1)
                candidate_target_hits.append(
                    {
                        "candidate_idx": int(cand_idx),
                        "candidate": int(candidate_row[cand_idx]),
                        "target_hit_indices": [
                            int(x) for x in hits[:4].tolist()
                        ],
                    }
                )

            topk_debug = []
            if has_topk_debug:
                assert row_base_topk_ids is not None
                assert row_base_topk_values is not None
                assert row_bias_topk_ids is not None
                assert row_bias_topk_values is not None
                assert row_final_topk_ids is not None
                assert row_final_topk_values is not None
                for step in range(draft_limit):
                    target_token = (
                        int(target_row[step])
                        if step < int(target_row.numel())
                        else None
                    )
                    candidate_token = (
                        int(candidate_row[step + 1])
                        if step + 1 < int(candidate_row.numel())
                        else None
                    )
                    base_ids = [int(x) for x in row_base_topk_ids[step].tolist()]
                    base_values = [
                        float(x) for x in row_base_topk_values[step].tolist()
                    ]
                    bias_ids = [int(x) for x in row_bias_topk_ids[step].tolist()]
                    bias_values = [
                        float(x) for x in row_bias_topk_values[step].tolist()
                    ]
                    final_ids = [int(x) for x in row_final_topk_ids[step].tolist()]
                    final_values = [
                        float(x) for x in row_final_topk_values[step].tolist()
                    ]
                    topk_debug.append(
                        {
                            "step": int(step),
                            "prev_token": int(row_prev[step]),
                            "candidate": candidate_token,
                            "target": target_token,
                            "base_top_ids": base_ids,
                            "base_top_logits": base_values,
                            "base_target_rank_in_top": (
                                base_ids.index(target_token)
                                if target_token in base_ids
                                else None
                            ),
                            "bias_top_ids": bias_ids,
                            "bias_top_logits": bias_values,
                            "bias_candidate_rank_in_top": (
                                bias_ids.index(candidate_token)
                                if candidate_token in bias_ids
                                else None
                            ),
                            "bias_target_rank_in_top": (
                                bias_ids.index(target_token)
                                if target_token in bias_ids
                                else None
                            ),
                            "final_top_ids": final_ids,
                            "final_top_logits": final_values,
                            "final_target_rank_in_top": (
                                final_ids.index(target_token)
                                if target_token in final_ids
                                else None
                            ),
                        }
                    )
            payload = {
                "layout": (
                    "candidate0 is verify anchor; markov_top1[i] is sampled from "
                    "block_hidden[i] and maps to candidate[i+1]; DeepSpec verify "
                    "stride is anchor + draft_width"
                ),
                "hidden_semantics": (
                    "post_norm_head_hidden; DeepSpec returns norm(hidden), while "
                    "vLLM DSV4 returns pre-norm hc_head hidden but computes base "
                    "logits as lm_head(norm(hidden))"
                ),
                "base_logits_semantics": "lm_head(post_norm_head_hidden)",
                "markov_semantics": (
                    "vanilla_markov_bias_depends_only_on_prev_token; hidden only "
                    "affects Markov refine through base logits"
                ),
                "post_norm_weight_l2": norm_weight_l2,
                "post_norm_weight_abs_mean": norm_weight_abs_mean,
                "base_top1_first": [int(x) for x in row_base.tolist()],
                "base_top1_logit_first": [
                    float(x) for x in row_base_top1_logit.tolist()
                ],
                "hidden_norm_first": [float(x) for x in row_hidden_norm.tolist()],
                "hidden_l2_ratio_to_norm_weight_first": [
                    float(x) for x in row_hidden_l2_ratio.tolist()
                ],
                "hidden_abs_mean_first": [
                    float(x) for x in row_hidden_abs_mean.tolist()
                ],
                "hidden_cos_adjacent_first": [
                    float(x) for x in row_hidden_cos_adjacent.tolist()
                ],
                "markov_top1_first": [
                    int(x) for x in row_markov[:draft_limit].tolist()
                ],
                "prev_tokens_first": [int(x) for x in row_prev.tolist()],
                "candidates_first": [
                    int(x) for x in candidate_row[:candidate_limit].tolist()
                ],
                "target_first": [int(x) for x in target_row[:candidate_limit].tolist()],
                "candidate_target_hits": candidate_target_hits,
                "logits_topk_first": topk_debug,
                "confidence_first": [
                    float(x) for x in confidence_row[:draft_limit].tolist()
                ],
            }
            if draft_width > 0:
                payload["markov0_eq_target0"] = bool(row_markov[0] == target_row[0])
                payload["candidate1_eq_markov0"] = (
                    bool(candidate_row[1] == row_markov[0])
                    if verify_stride > 1
                    else None
                )
            if draft_width > 1:
                payload["markov1_eq_target0"] = bool(row_markov[1] == target_row[0])
                payload["markov1_eq_target1"] = bool(row_markov[1] == target_row[1])
                payload["candidate2_eq_markov1"] = (
                    bool(candidate_row[2] == row_markov[1])
                    if verify_stride > 2
                    else None
                )
            return payload
        except Exception as e:
            return {"error": str(e)}

    def _build_accept_anomaly_logit_score_debug(
        self,
        *,
        row_idx: int,
        block_hidden: torch.Tensor,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
    ) -> Optional[list[dict]]:
        try:
            if (
                block_hidden is None
                or block_hidden.numel() == 0
                or candidates.numel() == 0
                or target_predict.numel() == 0
            ):
                return None
            row_idx = int(row_idx)
            if row_idx >= int(block_hidden.shape[0]):
                return None

            vocab_size = int(self._draft_inner.vocab_size)
            logits_processor = self.draft_model.logits_processor

            def _gather_full_vocab(logits_shard: torch.Tensor, head) -> torch.Tensor:
                def _reindex_sharded_vocab(logits: torch.Tensor) -> torch.Tensor:
                    mapping_fn = getattr(head, "get_sharded_to_full_mapping", None)
                    if mapping_fn is None:
                        return logits[..., :vocab_size]
                    mapping = mapping_fn()
                    if mapping is None:
                        return logits[..., :vocab_size]
                    cache_key = (id(head), int(logits.shape[-1]), logits.device)
                    mapping_tensor = self._vocab_shard_mapping_cache.get(cache_key)
                    if mapping_tensor is None or mapping_tensor.numel() != len(mapping):
                        mapping_tensor = torch.tensor(
                            mapping, dtype=torch.long, device=logits.device
                        )
                        self._vocab_shard_mapping_cache[cache_key] = mapping_tensor
                    return logits.index_select(-1, mapping_tensor)[..., :vocab_size]

                if logits_shard.shape[-1] >= vocab_size:
                    return _reindex_sharded_vocab(logits_shard)
                if getattr(head, "use_attn_tp_group", False):
                    group = get_attn_tp_group()
                    if group.world_size == 1:
                        return logits_shard[..., :vocab_size]
                    logits = group.all_gather(logits_shard, dim=-1)
                    return _reindex_sharded_vocab(logits)
                tp_size = get_tensor_model_parallel_world_size()
                if tp_size == 1:
                    return logits_shard[..., :vocab_size]
                logits = tensor_model_parallel_all_gather(logits_shard, dim=-1)
                return _reindex_sharded_vocab(logits)

            def _compute_full_vocab_logits(hidden_states: torch.Tensor, head):
                logits_shard = logits_processor._compute_lm_head(hidden_states, head)
                return _gather_full_vocab(logits_shard, head)

            with (
                self.draft_tp_context(self.draft_model_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                torch.inference_mode(),
            ):
                row_hidden = block_hidden[row_idx : row_idx + 1]
                base_logits = _compute_full_vocab_logits(
                    row_hidden, self.draft_model.lm_head
                )
                markov_head = self._draft_inner.markov_head
                candidate_row = candidates[row_idx]
                target_row = target_predict[row_idx]
                draft_width = min(
                    int(self.block_size),
                    int(base_logits.shape[1]),
                    int(target_row.numel()),
                    max(int(candidate_row.numel()) - 1, 0),
                )
                out = []
                prev_tokens = candidate_row[:draft_width].to(
                    device=block_hidden.device, dtype=torch.int64
                )
                for step in range(draft_width):
                    prev_token = prev_tokens[step].view(1)
                    candidate_token = int(candidate_row[step + 1].detach().cpu())
                    target_token = int(target_row[step].detach().cpu())
                    prev_embed = markov_head.get_prev_embeddings(prev_token)
                    bias_logits = _compute_full_vocab_logits(
                        prev_embed, markov_head.markov_w2
                    )[0]
                    base_step = base_logits[0, step]
                    final_step = base_step + bias_logits

                    def _score(logits: torch.Tensor, token: int) -> tuple[float, int]:
                        value = logits[int(token)]
                        rank = int((logits > value).sum().detach().cpu())
                        return float(value.detach().cpu()), rank

                    base_candidate, base_candidate_rank = _score(
                        base_step, candidate_token
                    )
                    base_target, base_target_rank = _score(base_step, target_token)
                    bias_candidate, bias_candidate_rank = _score(
                        bias_logits, candidate_token
                    )
                    bias_target, bias_target_rank = _score(bias_logits, target_token)
                    final_candidate, final_candidate_rank = _score(
                        final_step, candidate_token
                    )
                    final_target, final_target_rank = _score(final_step, target_token)
                    out.append(
                        {
                            "step": int(step),
                            "prev_token": int(prev_token.detach().cpu()[0]),
                            "candidate": candidate_token,
                            "target": target_token,
                            "base_candidate_logit": base_candidate,
                            "base_target_logit": base_target,
                            "base_candidate_rank": base_candidate_rank,
                            "base_target_rank": base_target_rank,
                            "bias_candidate_logit": bias_candidate,
                            "bias_target_logit": bias_target,
                            "bias_candidate_rank": bias_candidate_rank,
                            "bias_target_rank": bias_target_rank,
                            "final_candidate_logit": final_candidate,
                            "final_target_logit": final_target,
                            "final_candidate_rank": final_candidate_rank,
                            "final_target_rank": final_target_rank,
                            "final_margin_candidate_minus_target": (
                                final_candidate - final_target
                            ),
                        }
                    )
                return out
        except Exception as e:
            return [{"error": str(e)}]

    def _maybe_log_accept_anomaly(
        self,
        *,
        model_worker_batch: ScheduleBatch,
        draft_input: DSparkDraftInputV2,
        anchor_tokens: torch.Tensor,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
        commit_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        confidence: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        prefix_lens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        positions: torch.Tensor,
        block_hidden: Optional[torch.Tensor] = None,
        target_logits: Optional[torch.Tensor] = None,
        ignore_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not self._accept_anomaly_enabled or commit_lens.numel() == 0:
            return

        bs, block_size = candidates.shape
        commit_lens_cpu = commit_lens.detach().cpu().tolist()
        ignore_cpu = (
            ignore_mask.detach().cpu().tolist()
            if ignore_mask is not None and ignore_mask.numel() == bs
            else [False] * bs
        )
        if (
            not any(
                int(commit_len) <= 1 and not bool(ignore_cpu[i])
                for i, commit_len in enumerate(commit_lens_cpu)
            )
            and not self._accept_anomaly_histories
        ):
            return

        reqs = getattr(model_worker_batch, "reqs", None) or []
        req_pool_indices = model_worker_batch.req_pool_indices.detach().cpu().tolist()
        seq_lens = prefix_lens.detach().cpu().tolist()
        next_seq_lens = new_seq_lens.detach().cpu().tolist()
        candidate_cpu = candidates.detach().cpu()
        target_cpu = target_predict.detach().cpu()
        confidence_cpu = confidence.detach().cpu()
        next_bonus = bonus_tokens.detach().cpu().tolist()
        draft_anchor = (
            anchor_tokens.detach().cpu().tolist()
            if anchor_tokens.numel() == bs
            else None
        )
        input_bonus = (
            draft_input.bonus_tokens.detach().cpu().tolist()
            if draft_input.bonus_tokens.numel() == bs
            else None
        )
        future_indices = (
            draft_input.future_indices.detach().cpu().tolist()
            if draft_input.future_indices is not None
            and draft_input.future_indices.numel() == bs
            else None
        )
        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        sampling_is_all_greedy = (
            bool(getattr(sampling_info, "is_all_greedy"))
            if sampling_info is not None
            and hasattr(sampling_info, "is_all_greedy")
            else None
        )

        def _sampling_param_cpu(name: str):
            value = (
                getattr(sampling_info, name, None)
                if sampling_info is not None
                else None
            )
            if value is None:
                return None
            try:
                return value.detach().cpu().view(-1).tolist()
            except Exception:
                return None

        temperatures = _sampling_param_cpu("temperatures")
        top_ks = _sampling_param_cpu("top_ks")
        top_ps = _sampling_param_cpu("top_ps")
        verify_cache = verify_out_cache_loc.detach().cpu().view(bs, block_size)
        position_rows = positions.detach().cpu().view(bs, block_size)
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        max_req_len = int(req_to_token.shape[1])
        logical_probe = torch.stack(
            (
                torch.clamp(prefix_lens.to(torch.int64) - 1, min=0),
                prefix_lens.to(torch.int64),
                prefix_lens.to(torch.int64) + block_size - 1,
            ),
            dim=1,
        ).clamp(max=max_req_len - 1)
        req_pool_indices_gpu = model_worker_batch.req_pool_indices.to(
            device=req_to_token.device
        )
        req_to_token_probe = req_to_token[
            req_pool_indices_gpu[:, None], logical_probe.to(req_to_token.device)
        ]
        token_to_kv_pool = self.draft_model_runner.attn_backend.token_to_kv_pool
        translate_swa = getattr(token_to_kv_pool, "translate_loc_from_full_to_swa", None)
        if translate_swa is not None:
            try:
                probe_swa = translate_swa(req_to_token_probe).detach().cpu()
                verify_swa = translate_swa(verify_out_cache_loc).detach().cpu().view(
                    bs, block_size
                )
            except Exception:
                probe_swa = None
                verify_swa = None
        else:
            probe_swa = None
            verify_swa = None
        req_to_token_probe = req_to_token_probe.detach().cpu()
        logical_probe = logical_probe.detach().cpu()

        def _req_token_at(req, pos: int) -> Optional[int]:
            if req is None:
                return None
            try:
                origin = getattr(req, "origin_input_ids", []) or []
                output = getattr(req, "output_ids", []) or []
                origin_len = len(origin)
                if pos < 0:
                    return None
                if pos < origin_len:
                    return int(origin[pos])
                out_pos = pos - origin_len
                if out_pos < len(output):
                    return int(output[out_pos])
            except Exception:
                return None
            return None

        def _req_latest_token(req) -> Optional[int]:
            if req is None:
                return None
            try:
                output = getattr(req, "output_ids", []) or []
                if len(output) > 0:
                    return int(output[-1])
                origin = getattr(req, "origin_input_ids", []) or []
                if len(origin) > 0:
                    return int(origin[-1])
            except Exception:
                return None
            return None

        active_keys = set()
        for i, req_pool_idx in enumerate(req_pool_indices):
            req = reqs[i] if i < len(reqs) else None
            rid = getattr(req, "rid", None)
            key = (int(req_pool_idx), rid)
            active_keys.add(key)
            if bool(ignore_cpu[i]):
                self._accept_anomaly_histories.pop(key, None)
                self._accept_anomaly_streaks.pop(key, None)
                continue

            candidate_row = candidate_cpu[i]
            target_row = target_cpu[i]
            confidence_row = confidence_cpu[i]
            cache_row = verify_cache[i]
            commit_len = int(commit_lens_cpu[i])
            req_origin_len = (
                len(getattr(req, "origin_input_ids", []) or [])
                if req is not None
                else None
            )
            req_output_len = (
                len(getattr(req, "output_ids", []) or []) if req is not None else None
            )
            req_seq_len = (
                req_origin_len + req_output_len
                if req_origin_len is not None and req_output_len is not None
                else None
            )
            req_anchor_token = _req_token_at(req, int(seq_lens[i]) - 1)
            req_latest_token = _req_latest_token(req)
            draft_anchor_i = int(draft_anchor[i]) if draft_anchor is not None else None
            input_bonus_i = int(input_bonus[i]) if input_bonus is not None else None
            draft_block_metadata = self._last_draft_block_metadata_debug
            visible_block_sources = None
            valid_tail_sources = None
            if isinstance(draft_block_metadata, dict):
                block_locs_by_row = draft_block_metadata.get(
                    "draft_swa_valid_block_locs_by_row"
                )
                if isinstance(block_locs_by_row, list) and block_locs_by_row:
                    row_idx = min(i, len(block_locs_by_row) - 1)
                    row_locs = block_locs_by_row[row_idx]
                    if isinstance(row_locs, list):
                        visible_block_sources = self._visible_window_source_debug(
                            int(req_pool_idx), row_locs
                        )
                valid_edges = draft_block_metadata.get("draft_swa_valid_edges")
                if isinstance(valid_edges, dict):
                    tail_locs = valid_edges.get("first_valid_last8")
                    if isinstance(tail_locs, list):
                        valid_tail_sources = self._visible_window_source_debug(
                            int(req_pool_idx), tail_locs
                        )
            history = self._accept_anomaly_histories.setdefault(key, [])
            history.append(
                {
                    "rid": rid,
                    "req_pool_idx": int(req_pool_idx),
                    "future_idx": (
                        int(future_indices[i]) if future_indices is not None else None
                    ),
                    "seq_len": int(seq_lens[i]),
                    "new_seq_len": int(next_seq_lens[i]),
                    "is_all_greedy": sampling_is_all_greedy,
                    "temperature": (
                        float(temperatures[i])
                        if temperatures is not None and i < len(temperatures)
                        else None
                    ),
                    "top_k": (
                        int(top_ks[i])
                        if top_ks is not None and i < len(top_ks)
                        else None
                    ),
                    "top_p": (
                        float(top_ps[i])
                        if top_ps is not None and i < len(top_ps)
                        else None
                    ),
                    "kv_committed": (
                        int(getattr(req, "kv_committed_len"))
                        if req is not None and hasattr(req, "kv_committed_len")
                        else None
                    ),
                    "kv_allocated": (
                        int(getattr(req, "kv_allocated_len"))
                        if req is not None and hasattr(req, "kv_allocated_len")
                        else None
                    ),
                    "req_origin_len": req_origin_len,
                    "req_output_len": req_output_len,
                    "req_seq_len": req_seq_len,
                    "req_seq_len_minus_prefix": (
                        int(req_seq_len) - int(seq_lens[i])
                        if req_seq_len is not None
                        else None
                    ),
                    "req_anchor_token_at_prefix_minus_1": req_anchor_token,
                    "req_latest_token": req_latest_token,
                    "draft_anchor_token": draft_anchor_i,
                    "draft_anchor_matches_req_anchor": (
                        draft_anchor_i == req_anchor_token
                        if draft_anchor_i is not None and req_anchor_token is not None
                        else None
                    ),
                    "input_bonus": input_bonus_i,
                    "input_bonus_matches_req_anchor": (
                        input_bonus_i == req_anchor_token
                        if input_bonus_i is not None and req_anchor_token is not None
                        else None
                    ),
                    "input_bonus_matches_req_latest": (
                        input_bonus_i == req_latest_token
                        if input_bonus_i is not None and req_latest_token is not None
                        else None
                    ),
                    "candidate0": (
                        int(candidate_row[0]) if block_size > 0 else None
                    ),
                    "candidate0_matches_req_anchor": (
                        int(candidate_row[0]) == req_anchor_token
                        if block_size > 0 and req_anchor_token is not None
                        else None
                    ),
                    "candidate1": (
                        int(candidate_row[1]) if block_size > 1 else None
                    ),
                    "candidate2": (
                        int(candidate_row[2]) if block_size > 2 else None
                    ),
                    "target0": int(target_row[0]) if block_size > 0 else None,
                    "target1": int(target_row[1]) if block_size > 1 else None,
                    "target2": int(target_row[2]) if block_size > 2 else None,
                    "match1": (
                        bool(candidate_row[1] == target_row[0])
                        if block_size > 1
                        else None
                    ),
                    "match2": (
                        bool(candidate_row[2] == target_row[1])
                        if block_size > 2
                        else None
                    ),
                    "confidence0": (
                        float(confidence_row[0]) if block_size > 0 else None
                    ),
                    "confidence1": (
                        float(confidence_row[1]) if block_size > 1 else None
                    ),
                    "next_bonus": int(next_bonus[i]),
                    "accept_len": commit_len,
                    "verify_cache_first": int(cache_row[0]) if block_size > 0 else None,
                    "verify_cache_last": (
                        int(cache_row[-1]) if block_size > 0 else None
                    ),
                    "pos_first": int(position_rows[i][0]) if block_size > 0 else None,
                    "pos_last": int(position_rows[i][-1]) if block_size > 0 else None,
                    "logical_prev_cur_last": [
                        int(x) for x in logical_probe[i].tolist()
                    ],
                    "full_prev_cur_last": [
                        int(x) for x in req_to_token_probe[i].tolist()
                    ],
                    "swa_prev_cur_last": (
                        [int(x) for x in probe_swa[i].tolist()]
                        if probe_swa is not None
                        else None
                    ),
                    "verify_swa_first": (
                        int(verify_swa[i][0])
                        if verify_swa is not None and block_size > 0
                        else None
                    ),
                    "verify_swa_last": (
                        int(verify_swa[i][-1])
                        if verify_swa is not None and block_size > 0
                        else None
                    ),
                    "visible_block_sources": visible_block_sources,
                    "valid_tail_sources": valid_tail_sources,
                    "draft_block_metadata": draft_block_metadata,
                }
            )
            if len(history) > self._accept_anomaly_history_size:
                del history[: len(history) - self._accept_anomaly_history_size]

            if commit_len <= 1:
                streak = self._accept_anomaly_streaks.get(key, 0) + 1
                self._accept_anomaly_streaks[key] = streak
            else:
                self._accept_anomaly_streaks[key] = 0
                continue

            if (
                streak >= self._accept_anomaly_threshold
                and key not in self._accept_anomaly_dumped
                and self._accept_anomaly_dump_count < self._accept_anomaly_max_dumps
            ):
                self._accept_anomaly_dumped.add(key)
                self._accept_anomaly_dump_count += 1
                score_debug = None
                if block_hidden is not None:
                    score_debug = self._build_accept_anomaly_logit_score_debug(
                        row_idx=i,
                        block_hidden=block_hidden,
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                if not self._is_tp0():
                    continue
                kv_debug = self._build_accept_anomaly_kv_debug(
                    req_pool_idx=int(req_pool_idx),
                    prefix_len=int(seq_lens[i]),
                    block_size=int(self.block_size),
                )
                markov_debug = self._build_accept_anomaly_markov_debug(
                    row_idx=i,
                    candidates=candidate_cpu,
                    target_predict=target_cpu,
                    confidence=confidence_cpu,
                )
                if markov_debug is not None and score_debug is not None:
                    markov_debug["logit_score_debug"] = score_debug
                if (
                    markov_debug is not None
                    and target_logits is not None
                    and self._accept_anomaly_topk_enabled
                ):
                    try:
                        draft_width = int(self.block_size)
                        row_logits = target_logits.detach().view(
                            bs, int(self.verify_stride), -1
                        )[i, :draft_width]
                        topk_k = min(5, int(row_logits.shape[-1]))
                        topk_values, topk_ids = torch.topk(
                            row_logits, k=topk_k, dim=-1
                        )
                        target_row = target_predict[i, :draft_width]
                        candidate_row = candidates[i, 1 : 1 + draft_width]
                        markov_debug["target_logits_topk_first"] = [
                            {
                                "step": int(step),
                                "candidate": int(candidate_row[step]),
                                "target": int(target_row[step]),
                                "target_top_ids": [
                                    int(x) for x in topk_ids[step].detach().cpu().tolist()
                                ],
                                "target_top_logits": [
                                    float(x)
                                    for x in topk_values[step].detach().cpu().tolist()
                                ],
                            }
                            for step in range(draft_width)
                        ]
                    except Exception as e:
                        markov_debug["target_logits_topk_error"] = str(e)
                logger.warning(
                    "DSpark accept anomaly detected: dp_rank=%s tp_rank=%s "
                    "ep_rank=%s req_pool_idx=%s rid=%s zero_draft_streak=%s "
                    "history=%s",
                    self.dp_rank,
                    self.tp_rank,
                    self.moe_ep_rank,
                    int(req_pool_idx),
                    rid,
                    streak,
                    list(history),
                )
                logger.warning(
                    "DSpark accept anomaly KV debug: dp_rank=%s tp_rank=%s "
                    "ep_rank=%s req_pool_idx=%s rid=%s kv_debug=%s",
                    self.dp_rank,
                    self.tp_rank,
                    self.moe_ep_rank,
                    int(req_pool_idx),
                    rid,
                    kv_debug,
                )
                logger.warning(
                    "DSpark accept anomaly Markov debug: dp_rank=%s tp_rank=%s "
                    "ep_rank=%s req_pool_idx=%s rid=%s markov_debug=%s",
                    self.dp_rank,
                    self.tp_rank,
                    self.moe_ep_rank,
                    int(req_pool_idx),
                    rid,
                    markov_debug,
                )

        stale_keys = set(self._accept_anomaly_histories) - active_keys
        for key in stale_keys:
            self._accept_anomaly_histories.pop(key, None)
            self._accept_anomaly_streaks.pop(key, None)

    def _clear_unaccepted_c128_draft_states(
        self,
        *,
        batch: ScheduleBatch,
        prefix_lens: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        allocator = getattr(batch, "token_to_kv_pool_allocator", None)
        if allocator is None:
            return
        kvcache = allocator.get_kvcache()
        clear_unaccepted_c128 = getattr(
            kvcache, "clear_unaccepted_c128_draft_states", None
        )
        if clear_unaccepted_c128 is None:
            return
        clear_unaccepted_c128(
            batch.req_pool_indices,
            prefix_lens,
            commit_lens,
            int(self.block_size),
        )

    def _make_next_draft_input_prefill(
        self,
        *,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
        prefill_tail_hidden_states: Optional[torch.Tensor] = None,
        prefill_tail_valid_mask: Optional[torch.Tensor] = None,
    ) -> DSparkDraftInputV2:
        if prefill_tail_hidden_states is None:
            prefill_tail_hidden_states = torch.empty(
                (0, 0, 0), dtype=torch.float16, device=bonus_tokens.device
            )
        if prefill_tail_valid_mask is None:
            prefill_tail_valid_mask = torch.empty(
                (0, 0), dtype=torch.bool, device=bonus_tokens.device
            )
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
            prefill_tail_hidden_states=prefill_tail_hidden_states,
            prefill_tail_valid_mask=prefill_tail_valid_mask,
            transfer_warmup_rounds=torch.zeros_like(seq_lens, dtype=torch.int32),
        )

    def _get_transfer_warmup_rounds(
        self, draft_input: DSparkDraftInputV2, bs: int, device: torch.device
    ) -> torch.Tensor:
        rounds = draft_input.transfer_warmup_rounds
        if rounds.numel() == bs:
            return rounds.to(device=device, dtype=torch.int32)
        return torch.zeros((bs,), dtype=torch.int32, device=device)

    def _get_decode_anchor_tokens(
        self,
        *,
        batch: ScheduleBatch,
        prefix_lens: torch.Tensor,
        fallback_tokens: torch.Tensor,
        bs: int,
        device: torch.device,
    ) -> torch.Tensor:
        reqs = getattr(batch, "reqs", None) or []
        seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
        try:
            if seq_lens_cpu is not None and len(seq_lens_cpu) == bs:
                prefix_lens_list = [int(x) for x in seq_lens_cpu]
            else:
                prefix_lens_list = [int(x) for x in prefix_lens.detach().cpu().tolist()]
        except Exception:
            prefix_lens_list = []
        fallback_list = (
            fallback_tokens.detach().cpu().tolist()
            if fallback_tokens.numel() == bs
            else None
        )
        anchors = []
        for i in range(bs):
            token = None
            req = reqs[i] if i < len(reqs) else None
            prefix_len_i = (
                int(prefix_lens_list[i]) if i < len(prefix_lens_list) else None
            )
            if req is not None:
                try:
                    origin = getattr(req, "origin_input_ids", []) or []
                    output = getattr(req, "output_ids", []) or []
                    origin_len = len(origin)
                    req_seq_len = origin_len + len(output)
                    if (
                        prefix_len_i is not None
                        and req_seq_len < prefix_len_i
                        and fallback_list is not None
                    ):
                        token = int(fallback_list[i])
                    else:
                        pos = int(prefix_len_i) - 1 if prefix_len_i is not None else -1
                        if 0 <= pos < origin_len:
                            token = int(origin[pos])
                        else:
                            out_pos = pos - origin_len
                            if 0 <= out_pos < len(output):
                                token = int(output[out_pos])
                except Exception:
                    token = None
                if token is None:
                    try:
                        output = getattr(req, "output_ids", []) or []
                        if len(output) > 0:
                            token = int(output[-1])
                        else:
                            origin = getattr(req, "origin_input_ids", []) or []
                            if len(origin) > 0:
                                token = int(origin[-1])
                    except Exception:
                        token = None
            if token is None and fallback_list is not None:
                token = int(fallback_list[i])
            if token is None:
                token = int(self.noise_token_id)
            anchors.append(token)
        return torch.tensor(anchors, dtype=torch.int64, device=device)

    def _make_next_draft_input_decode(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
        transfer_warmup_rounds: Optional[torch.Tensor] = None,
    ) -> DSparkDraftInputV2:
        if transfer_warmup_rounds is None:
            transfer_warmup_rounds = torch.zeros_like(new_seq_lens, dtype=torch.int32)
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
            transfer_warmup_rounds=transfer_warmup_rounds.to(dtype=torch.int32),
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and self._is_tp0()
            and not getattr(self, "_warned_sampling", False)
        ):
            self._warned_sampling = True
            logger.warning(
                "DSpark verifies greedily; temperature>0 requests are served with "
                "greedy verification. Rejection-sampling support is a follow-up."
            )

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            return self._forward_prefill(model_worker_batch, on_publish)

        return self._forward_decode(model_worker_batch, on_publish)

    def _forward_prefill(
        self, model_worker_batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(model_worker_batch)

        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = model_worker_batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        device = (
            next_token_ids.device
            if next_token_ids is not None
            else model_worker_batch.seq_lens.device
        )
        local_extend_tokens = int(getattr(model_worker_batch, "extend_num_tokens", 0))
        if local_extend_tokens == 0 and model_worker_batch.input_ids is not None:
            local_extend_tokens = int(model_worker_batch.input_ids.numel())
        if local_extend_tokens == 0:
            if next_token_ids is None:
                next_token_ids = torch.empty(
                    (0,), dtype=torch.int64, device=model_worker_batch.seq_lens.device
                )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = None
            batch_output.next_draft_input = self._make_next_draft_input_prefill(
                bonus_tokens=next_token_ids,
                seq_lens=model_worker_batch.seq_lens,
                cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            batch_output.next_draft_input.verify_done = verify_done
            return batch_output

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DSpark target layers configured."
            )
        if model_worker_batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        extend_lens = model_worker_batch.extend_lens
        prefix_lens = model_worker_batch.prefix_lens
        if extend_lens is None or prefix_lens is None:
            reqs = getattr(model_worker_batch, "reqs", None) or []
            if len(reqs) != len(model_worker_batch.seq_lens):
                raise RuntimeError(
                    "DSpark expected extend_lens / prefix_lens in extend mode, "
                    "and could not rebuild them from batch requests."
                )
            prefix_lens = [len(req.prefix_indices) for req in reqs]
            extend_lens = [req.extend_range.length for req in reqs]

        ctx_lens = torch.tensor(extend_lens, dtype=torch.int32, device=device)
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        # For chunked prefill, the draft-context RoPE start is the logical start
        # of this chunk. Derive it from post-prefill seq_lens so context KV
        # materialization stays correct even if prefix_lens only reflects a
        # cache-prefix notion on a specialized scheduling path.
        draft_seq_lens = (
            model_worker_batch.seq_lens.to(device=device, dtype=torch.int32) - ctx_lens
        ).clamp_min_(0)
        if self._accept_anomaly_enabled and not torch.equal(
            draft_seq_lens, prefix_lens_tensor
        ):
            if self._is_tp0():
                logger.warning(
                    "DSpark prefill materialize prefix mismatch: prefix_lens=%s "
                    "seq_lens_minus_extend=%s extend_lens=%s seq_lens=%s",
                    [int(x) for x in prefix_lens_tensor.detach().cpu().tolist()],
                    [int(x) for x in draft_seq_lens.detach().cpu().tolist()],
                    [int(x) for x in ctx_lens.detach().cpu().tolist()],
                    [
                        int(x)
                        for x in model_worker_batch.seq_lens.detach().cpu().tolist()
                    ],
                )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(extend_lens)),
        )

        extend_lens_list = [int(x) for x in extend_lens]
        if not extend_lens_list:
            raise RuntimeError("DSpark prefill expected non-empty extend_lens.")

        draft_forward_batch = self._make_draft_prefill_forward_batch_for_materialize(
            model_worker_batch
        )
        main_x = self.draft_model.project_main_hidden(logits_output.hidden_states)
        if self._accept_anomaly_enabled:
            offset = 0
            for i, extend_len in enumerate(extend_lens_list):
                next_offset = offset + int(extend_len)
                if extend_len > 0:
                    take_start = max(offset, next_offset - 8)
                    req_pool_idx = int(model_worker_batch.req_pool_indices[i])
                    self._boundary_debug_by_req_pool[req_pool_idx] = {}
                    self._target_aux_debug_by_req_pool[req_pool_idx] = {}
                    self._record_boundary_debug(
                        req_pool_idx=req_pool_idx,
                        stage="prefill_tail_source",
                        positions=positions[take_start:next_offset],
                        cache_locs=model_worker_batch.out_cache_loc[
                            take_start:next_offset
                        ],
                        hidden_rows=main_x[take_start:next_offset],
                        extra={
                            "extend_len": int(extend_len),
                            "post_seq_len": int(model_worker_batch.seq_lens[i]),
                        },
                    )
                    self._record_target_aux_debug(
                        req_pool_idx=req_pool_idx,
                        stage="prefill_tail_aux",
                        raw_hidden_rows=logits_output.hidden_states[
                            take_start:next_offset
                        ],
                        projected_rows=main_x[take_start:next_offset],
                    )
                offset = next_offset
        self._materialize_main_hidden_to_draft_state(
            main_hidden=main_x,
            cache_loc=model_worker_batch.out_cache_loc,
            positions=positions,
            draft_forward_batch=draft_forward_batch,
            projected=True,
        )
        prefill_tail_hidden, prefill_tail_mask = self._pack_prefill_tail_hidden(
            hidden=main_x,
            extend_lens=extend_lens_list,
        )

        logits_output.hidden_states = None

        batch_output.next_draft_input = self._make_next_draft_input_prefill(
            bonus_tokens=next_token_ids,
            seq_lens=model_worker_batch.seq_lens,
            cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            prefill_tail_hidden_states=prefill_tail_hidden,
            prefill_tail_valid_mask=prefill_tail_mask,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        batch_output.next_draft_input.verify_done = verify_done
        return batch_output

    def _forward_decode(
        self, model_worker_batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DSparkDraftInputV2.create_idle_input(
                device=self.device
            )
        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DSparkDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DSparkDraftInputV2 state on the running batch."
            )

        participates_in_dp_decode = (
            self.server_args.enable_dp_attention
            and model_worker_batch.forward_mode.is_idle()
            and model_worker_batch.global_num_tokens is not None
            and any(int(x) > 0 for x in model_worker_batch.global_num_tokens)
        )
        if model_worker_batch.forward_mode.is_idle() and not participates_in_dp_decode:
            return self._forward_idle(on_publish)
        dp_decode_global_num_tokens = self._get_dp_decode_global_num_tokens(
            model_worker_batch
        )

        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        device = self.device
        bs = len(model_worker_batch.seq_lens)
        block_size = int(self.block_size)
        verify_stride = int(self.verify_stride)
        prefix_lens = model_worker_batch.seq_lens
        anchor_lens = (prefix_lens.to(torch.int64) - 1).clamp_min(0)
        req_pool_indices = model_worker_batch.req_pool_indices

        anchor_tokens = self._get_decode_anchor_tokens(
            batch=model_worker_batch,
            prefix_lens=prefix_lens,
            fallback_tokens=draft_input.bonus_tokens,
            bs=bs,
            device=device,
        )
        block_ids = torch.full(
            (bs, block_size), self.noise_token_id, dtype=torch.int64, device=device
        )
        if anchor_tokens.numel() == bs:
            block_ids[:, 0].copy_(anchor_tokens.view(-1))

        positions_2d = anchor_lens.unsqueeze(1) + self._block_pos_offsets
        positions = positions_2d.reshape(-1).to(torch.int64)
        verify_positions_2d = anchor_lens.unsqueeze(1) + self._verify_pos_offsets
        verify_positions = verify_positions_2d.reshape(-1).to(torch.int64)

        end_offset = anchor_lens + verify_stride
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=anchor_lens,
            end_offset=end_offset,
            batch_size=bs,
            draft_token_num=verify_stride,
            device=device,
        )
        draft_out_cache_loc = verify_out_cache_loc.view(bs, verify_stride)[
            :, :block_size
        ].reshape(-1)
        self._materialize_disagg_prefill_hidden_to_draft_state(
            draft_input=draft_input,
            batch=model_worker_batch,
            prefix_lens=prefix_lens,
        )
        self._materialize_prefill_tail_hidden_to_draft_state(
            draft_input=draft_input,
            batch=model_worker_batch,
            prefix_lens=prefix_lens,
        )

        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            block_hidden, draft_forward_batch = self._run_draft_block(
                batch=model_worker_batch,
                bs=bs,
                block_ids=block_ids,
                positions=positions,
                verify_out_cache_loc=draft_out_cache_loc,
                seq_lens_for_metadata=anchor_lens,
                dp_decode_global_num_tokens=dp_decode_global_num_tokens,
            )
            self._record_draft_block_write_sources(
                req_pool_indices=req_pool_indices,
                draft_out_cache_loc=draft_out_cache_loc,
            )

            candidates, confidence = self._refine_block_markov(
                block_hidden=block_hidden,
                anchor_tokens=anchor_tokens,
                output_bs=bs,
            )

        verify_input = DSparkVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=verify_positions,
            draft_token_num=verify_stride,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        model_worker_batch.out_cache_loc = verify_out_cache_loc
        if participates_in_dp_decode:
            model_worker_batch.forward_mode = ForwardMode.DECODE
        original_global_num_tokens = model_worker_batch.global_num_tokens
        original_global_num_tokens_for_logprob = (
            model_worker_batch.global_num_tokens_for_logprob
        )
        original_seq_lens = model_worker_batch.seq_lens
        original_seq_lens_cpu = model_worker_batch.seq_lens_cpu
        original_seq_lens_sum = model_worker_batch.seq_lens_sum
        if dp_decode_global_num_tokens is not None:
            model_worker_batch.global_num_tokens = dp_decode_global_num_tokens
            if original_global_num_tokens_for_logprob is not None:
                model_worker_batch.global_num_tokens_for_logprob = (
                    dp_decode_global_num_tokens
                )
        try:
            model_worker_batch.seq_lens = anchor_lens.to(dtype=original_seq_lens.dtype)
            model_worker_batch.seq_lens_cpu = model_worker_batch.seq_lens.detach().cpu()
            model_worker_batch.seq_lens_sum = int(
                model_worker_batch.seq_lens_cpu.sum().item()
            )
            verify_forward_batch, _ = verify_input.prepare_for_verify(
                model_worker_batch, self.target_worker
            )
        finally:
            model_worker_batch.global_num_tokens = original_global_num_tokens
            model_worker_batch.global_num_tokens_for_logprob = (
                original_global_num_tokens_for_logprob
            )
            model_worker_batch.seq_lens = original_seq_lens
            model_worker_batch.seq_lens_cpu = original_seq_lens_cpu
            model_worker_batch.seq_lens_sum = original_seq_lens_sum
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, verify_stride
        )

        new_seq_lens = None
        if bs == 0:
            bonus_tokens = torch.empty((0,), dtype=torch.int64, device=device)
            commit_lens = torch.empty((0,), dtype=torch.int32, device=device)
            out_tokens = torch.empty(
                (0, verify_stride), dtype=torch.int64, device=device
            )
        elif self._use_triton_accept_bonus:
            try:
                (
                    commit_lens,
                    bonus_tokens,
                    out_tokens,
                    new_seq_lens,
                ) = self._next_accept_bonus_buffers(bs)
                _compute_dspark_accept_bonus_triton_unchecked(
                    candidates=candidates,
                    target_top1=target_predict,
                    confidence=confidence,
                    commit_lens_out=commit_lens,
                    bonus_ids_out=bonus_tokens,
                    out_tokens_out=out_tokens,
                    prefix_lens=prefix_lens,
                    new_seq_lens_out=new_seq_lens,
                    confidence_threshold=self.confidence_threshold,
                )
            except Exception as e:
                self._use_triton_accept_bonus = False
                if self._is_tp0():
                    logger.warning(
                        "DSPARK Triton accept/bonus failed; "
                        "falling back to eager path: %s",
                        e,
                    )
                commit_lens, bonus_tokens, out_tokens = self._compute_accept_bonus_eager(
                    candidates=candidates,
                    target_predict=target_predict,
                    confidence=confidence,
                )
        else:
            commit_lens, bonus_tokens, out_tokens = self._compute_accept_bonus_eager(
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        transfer_warmup_rounds = self._get_transfer_warmup_rounds(
            draft_input, bs, device
        )
        transfer_warmup_mask = transfer_warmup_rounds > 0
        next_transfer_warmup_rounds = torch.clamp(
            transfer_warmup_rounds - 1, min=0
        )
        if transfer_warmup_mask.any():
            target0 = target_predict[:, 0]
            commit_lens = torch.where(
                transfer_warmup_mask,
                torch.ones_like(commit_lens),
                commit_lens,
            )
            bonus_tokens = torch.where(transfer_warmup_mask, target0, bonus_tokens)
            new_seq_lens = torch.where(
                transfer_warmup_mask,
                prefix_lens + 1,
                new_seq_lens,
            )
            out_tokens[transfer_warmup_mask] = 0
            out_tokens[transfer_warmup_mask, 0] = target0[transfer_warmup_mask]

        hidden = logits_output.hidden_states
        main_x_flat = None
        commit_mask = None
        if hidden is None and bs > 0:
            raise RuntimeError(
                "DSpark verify requires target main_hidden states, but got None."
            )
        if bs > 0:
            hidden = hidden.view(bs, verify_stride, -1)
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            main_x_flat = self.draft_model.project_main_hidden(hidden_flat)
            commit_mask_2d = (
                self._verify_pos_offsets.unsqueeze(0)
                < commit_lens.unsqueeze(1).to(torch.int64)
            )
            commit_mask = commit_mask_2d.reshape(-1)
            if self._accept_anomaly_enabled:
                cache_locs_2d = verify_out_cache_loc.view(bs, verify_stride)
                positions_2d = verify_positions.view(bs, verify_stride)
                main_x_2d = main_x_flat.view(bs, verify_stride, -1)
                for i in range(bs):
                    row_valid = commit_mask_2d[i]
                    if not row_valid.any():
                        continue
                    take = torch.nonzero(row_valid, as_tuple=False).view(-1)[-8:]
                    self._record_boundary_debug(
                        req_pool_idx=int(req_pool_indices[i]),
                        stage="decode_verify_write",
                        positions=positions_2d[i, take],
                        cache_locs=cache_locs_2d[i, take],
                        hidden_rows=main_x_2d[i, take],
                        extra={
                            "commit_len": int(commit_lens[i]),
                            "prefix_len": int(prefix_lens[i]),
                            "new_seq_len": int(new_seq_lens[i]),
                        },
                    )
                    self._record_target_aux_debug(
                        req_pool_idx=int(req_pool_indices[i]),
                        stage="decode_verify_aux",
                        raw_hidden_rows=hidden[i, take],
                        projected_rows=main_x_2d[i, take],
                    )
        self._maybe_log_accept_anomaly(
            model_worker_batch=model_worker_batch,
            draft_input=draft_input,
            anchor_tokens=anchor_tokens,
            candidates=candidates,
            target_predict=target_predict,
            commit_lens=commit_lens,
            bonus_tokens=bonus_tokens,
            confidence=confidence,
            verify_out_cache_loc=verify_out_cache_loc,
            prefix_lens=prefix_lens,
            new_seq_lens=new_seq_lens,
            positions=verify_positions,
            block_hidden=block_hidden,
            target_logits=logits_output.next_token_logits,
            ignore_mask=transfer_warmup_mask,
        )
        if on_publish is not None:
            on_publish(new_seq_lens)

        if bs > 0:
            self._materialize_main_hidden_to_draft_state(
                main_hidden=main_x_flat,
                cache_loc=verify_out_cache_loc[commit_mask],
                positions=verify_positions[commit_mask],
                draft_forward_batch=draft_forward_batch,
                kv_main_hidden=main_x_flat[commit_mask],
                projected=True,
            )
            self._clear_unaccepted_c128_draft_states(
                batch=model_worker_batch,
                prefix_lens=prefix_lens,
                commit_lens=commit_lens,
            )

        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
            transfer_warmup_rounds=next_transfer_warmup_rounds,
        )
        next_draft_input.carry_prepare_buffers_from(draft_input)
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=verify_stride,
            new_seq_lens=new_seq_lens,
        )

    def _forward_idle(self, on_publish) -> GenerationBatchResult:
        empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
        empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=empty_ids,
            accept_lens=empty_lens,
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_stride),
            new_seq_lens=next_draft_input.new_seq_lens,
        )
