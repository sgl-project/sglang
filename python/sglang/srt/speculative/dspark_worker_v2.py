import contextlib
import logging
import os
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

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
from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.dspark_info import (
    DSparkDraftBlockInput,
    DSparkDraftExtendInput,
    DSparkDraftInputV2,
    DSparkVerifyInput,
)
from sglang.srt.speculative.spec_utils import draft_tp_context, spec_stage_span
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

        self.draft_model.model.embed_tokens.weight = (
            self.target_worker.model_runner.model.model.embed_tokens.weight
        )
        self.draft_model.lm_head.weight = (
            self.target_worker.model_runner.model.lm_head.weight
        )

        self.block_size = int(server_args.speculative_num_draft_tokens)
        model_block_size = int(getattr(self.draft_model, "block_size", self.block_size))
        if model_block_size != self.block_size:
            logger.warning(
                "DSpark block size mismatch: using speculative_num_draft_tokens=%s "
                "but draft model block_size=%s.",
                self.block_size,
                model_block_size,
            )
        self.speculative_num_draft_tokens = int(self.block_size)

        self.noise_token_id = int(self._draft_inner.noise_token_id)
        self.markov_rank = int(self._draft_inner.markov_rank)
        self.num_dspark_layers = int(self.draft_model.num_dspark_layers)
        self.confidence_threshold = float(
            server_args.speculative_dspark_confidence_threshold
        )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self._use_triton_accept_bonus = is_cuda() or is_hip()
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []
        self._markov_refine_buffer_cap: int = 0
        self._markov_candidates_buf: Optional[torch.Tensor] = None
        self._markov_embeds_buf: Optional[torch.Tensor] = None
        self.draft_attn_backend = None
        self.draft_extend_attn_backend = None
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)
        self._accept_anomaly_enabled = _env_flag("SGLANG_DSPARK_DEBUG_ACCEPT", True)
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
        self._stacked_wqkv_fp8_proj = None
        self._stacked_wqkv_kv_offsets: list[tuple[int, int]] = []
        self._stacked_wqkv_out_sizes: list[int] = []
        self._init_fp8_wqkv_stack()

        if self.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. model=%s, block_size=%s, "
                "num_dspark_layers=%s, noise_token_id=%s, markov_rank=%s, "
                "confidence_threshold=%s",
                self.draft_model.__class__.__name__,
                self.block_size,
                self.num_dspark_layers,
                self.noise_token_id,
                self.markov_rank,
                self.confidence_threshold,
            )

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

        if self.tp_rank == 0:
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
        if self.tp_rank == 0:
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
            self._draft_worker.init_cuda_graphs()

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
            main_x = self.draft_model.project_main_hidden(main_hidden)
            if self._stacked_wqkv_fp8_proj is None:
                for layer in self._draft_inner.layers:
                    layer.self_attn.kv_from_hidden(
                        main_x, positions, cache_loc, attn_backend
                    )
            else:
                stacked_out = self._stacked_wqkv_fp8_proj.quant_method.apply(
                    self._stacked_wqkv_fp8_proj,
                    main_x,
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
                main_x = self.draft_model.project_main_hidden(main_hidden)
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
    ) -> None:
        try:
            self._materialize_main_hidden_to_draft_compressors(
                main_hidden=main_hidden,
                draft_forward_batch=draft_forward_batch,
            )
        except Exception as e:
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
        participates_in_dp_decode = (
            self.server_args.enable_dp_attention
            and dp_decode_global_num_tokens is not None
            and any(int(x) > 0 for x in dp_decode_global_num_tokens)
        )
        if main_hidden.numel() == 0 and not participates_in_dp_decode:
            return
        if not use_draft_extend_v2:
            raise RuntimeError(
                "DSpark bootstrap forward only supports DRAFT_EXTEND_V2."
            )

        bs = int(len(seq_lens if seq_lens is not None else batch.seq_lens))
        num_tokens = int(input_ids.numel())
        if bs == 0:
            if num_tokens != 0 or main_hidden.shape[0] != 0:
                raise RuntimeError(
                    "DSpark idle draft-extend bootstrap expected empty local "
                    f"inputs: hidden={tuple(main_hidden.shape)} "
                    f"input_tokens={num_tokens}"
                )
            if not participates_in_dp_decode:
                return
        elif num_tokens == 0:
            return
        if main_hidden.shape[0] != num_tokens:
            raise RuntimeError(
                "DSpark draft bootstrap hidden/input size mismatch: "
                f"hidden={tuple(main_hidden.shape)} input_tokens={num_tokens}"
            )
        if bs > 0 and num_tokens % bs != 0:
            raise RuntimeError(
                "DSpark draft-extend-v2 bootstrap requires fixed tokens per "
                f"request: num_tokens={num_tokens} bs={bs}"
            )
        if bs > 0:
            inferred_tokens_per_req = num_tokens // bs
            if int(num_tokens_per_req) != inferred_tokens_per_req:
                num_tokens_per_req = inferred_tokens_per_req

        old_input_ids = batch.input_ids
        old_out_cache_loc = batch.out_cache_loc
        old_spec_info = batch.spec_info
        old_forward_mode = batch.forward_mode
        old_capture_hidden_mode = batch.capture_hidden_mode
        old_seq_lens = batch.seq_lens
        old_seq_lens_cpu = batch.seq_lens_cpu
        old_seq_lens_sum = batch.seq_lens_sum
        old_req_pool_indices = batch.req_pool_indices
        old_prefix_lens = batch.prefix_lens
        old_extend_lens = batch.extend_lens
        old_extend_num_tokens = batch.extend_num_tokens
        old_global_num_tokens = batch.global_num_tokens
        old_global_num_tokens_for_logprob = batch.global_num_tokens_for_logprob
        old_attn_backend = self.draft_model_runner.attn_backend
        try:
            if self.draft_extend_attn_backend is None:
                raise RuntimeError(
                    "DSpark draft_extend_attn_backend is not initialized."
                )
            batch.out_cache_loc = out_cache_loc.to(device=self.device)
            if seq_lens is not None:
                batch.seq_lens = seq_lens.to(
                    device=self.device, dtype=old_seq_lens.dtype
                )
                batch.seq_lens_cpu = batch.seq_lens.detach().cpu()
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
            if req_pool_indices is not None:
                batch.req_pool_indices = req_pool_indices.to(
                    device=self.device, dtype=old_req_pool_indices.dtype
                )
            if dp_decode_global_num_tokens is not None:
                batch.global_num_tokens = dp_decode_global_num_tokens
                batch.global_num_tokens_for_logprob = dp_decode_global_num_tokens

            draft_extend_input = DSparkDraftExtendInput(
                hidden_states=main_hidden.to(self.device, non_blocking=True),
                positions=positions.to(device=self.device),
                num_tokens_per_req=num_tokens_per_req,
                num_tokens_for_logprob_per_req=num_tokens_per_req,
            )
            predict = input_ids.to(device=self.device, dtype=torch.int64)
            self.draft_model_runner.attn_backend = self.draft_extend_attn_backend
            with (
                self.draft_tp_context(self.draft_model_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
                torch.inference_mode(),
            ):
                with self.plan_stream_ctx:
                    forward_batch = self.prepare_for_draft_extend(
                        draft_extend_input,
                        batch,
                        predict,
                        int(num_tokens_per_req),
                        self.draft_model_runner,
                        None,
                    )
                forward_batch.return_logprob = False
                forward_batch.lora_ids = [None] * forward_batch.batch_size
                if self.plan_stream:
                    torch.get_device_module(self.device).current_stream().wait_stream(
                        self.plan_stream
                    )
                self.draft_model_runner.forward(forward_batch)
        finally:
            batch.input_ids = old_input_ids
            batch.out_cache_loc = old_out_cache_loc
            batch.spec_info = old_spec_info
            batch.forward_mode = old_forward_mode
            batch.capture_hidden_mode = old_capture_hidden_mode
            batch.seq_lens = old_seq_lens
            batch.seq_lens_cpu = old_seq_lens_cpu
            batch.seq_lens_sum = old_seq_lens_sum
            batch.req_pool_indices = old_req_pool_indices
            batch.prefix_lens = old_prefix_lens
            batch.extend_lens = old_extend_lens
            batch.extend_num_tokens = old_extend_num_tokens
            batch.global_num_tokens = old_global_num_tokens
            batch.global_num_tokens_for_logprob = old_global_num_tokens_for_logprob
            self.draft_model_runner.attn_backend = old_attn_backend

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
            logger.warning(
                "Skip DSpark PD hidden bootstrap due to shape mismatch: "
                "hidden_shape=%s bs=%s",
                tuple(hidden.shape),
                bs,
            )
            return
        expected_hidden_size = self._get_target_aux_hidden_size()
        if expected_hidden_size and hidden.shape[-1] != expected_hidden_size:
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
        dp_decode_global_num_tokens: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, object]:
        draft_block_spec_info = DSparkDraftBlockInput(
            draft_token=block_ids.reshape(-1),
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        draft_forward_batch = draft_block_spec_info.prepare_for_draft_block(
            batch=batch,
            draft_model_runner=self.draft_model_runner,
            out_cache_loc=verify_out_cache_loc,
            dp_decode_global_num_tokens=dp_decode_global_num_tokens,
        )

        from sglang.srt.layers.attention import deepseek_v4_backend as _dsv4_be

        _dsv4_be._DSPARK_BLOCK_FULL_ATTN = int(self.block_size)
        try:
            with torch.inference_mode():
                draft_runner_out = self.draft_model_runner.forward(draft_forward_batch)
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
            (new_cap, int(self.block_size)), dtype=torch.int64, device=device
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
        bonus_tokens: torch.Tensor,
        output_bs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = int(block_hidden.shape[0])
        output_bs = bs if output_bs is None else int(output_bs)
        block_size = int(self.block_size)
        if bs == 0:
            empty_tokens = torch.empty(
                (output_bs, block_size), dtype=torch.int64, device=block_hidden.device
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

        vocab_size = int(self._draft_inner.vocab_size)

        def _gather_full_vocab(logits_shard: torch.Tensor, head) -> torch.Tensor:
            if logits_shard.shape[-1] >= vocab_size:
                return logits_shard[..., :vocab_size]
            if getattr(head, "use_attn_tp_group", False):
                group = get_attn_tp_group()
                if group.world_size == 1:
                    return logits_shard
                return group.all_gather(logits_shard, dim=-1)[..., :vocab_size]
            tp_size = get_tensor_model_parallel_world_size()
            if tp_size == 1:
                return logits_shard
            return tensor_model_parallel_all_gather(logits_shard, dim=-1)[
                ..., :vocab_size
            ]

        if bonus_tokens.numel() == bs:
            first_tokens = bonus_tokens.view(-1).to(torch.int64)
        else:
            first_tokens = torch.full(
                (bs,), self.noise_token_id, dtype=torch.int64, device=block_hidden.device
            )
        candidates[:, 0].copy_(first_tokens)

        with torch.inference_mode():
            base_logits = _gather_full_vocab(
                F.linear(block_hidden, lm_head.weight), lm_head
            )
            prev_tokens = candidates[:, 0]
            for i in range(block_size):
                prev_embed = markov_head.get_prev_embeddings(prev_tokens)
                markov_embeds[:, i].copy_(prev_embed)
                bias = _gather_full_vocab(
                    markov_head.project_bias(prev_embed), markov_head.markov_w2
                )
                bias.add_(base_logits[:, i])
                next_tokens = torch.argmax(bias, dim=-1)
                if i + 1 < block_size:
                    candidates[:, i + 1].copy_(next_tokens)
                prev_tokens = next_tokens

            confidence = confidence_head(block_hidden, markov_embeds)

        return candidates[:output_bs], confidence[:output_bs]

    def _confident_prefix(self, confidence: torch.Tensor) -> torch.Tensor:
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
        block_size = int(self.block_size)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
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
        bs, block_size = candidates.shape
        correct_len, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        confident_prefix = self._confident_prefix(confidence)
        correct_len = torch.minimum(
            correct_len.to(torch.int64), confident_prefix.to(torch.int64)
        )
        bonus_tokens = target_predict.gather(1, correct_len.unsqueeze(1)).squeeze(1)
        commit_lens = correct_len.to(torch.int32) + 1

        out_tokens = torch.empty(
            (bs, block_size), dtype=torch.int64, device=candidates.device
        )
        if block_size > 1:
            out_tokens[:, : block_size - 1].copy_(candidates[:, 1:])
        out_tokens[:, block_size - 1].fill_(0)
        out_tokens.scatter_(
            1,
            correct_len.unsqueeze(1),
            bonus_tokens.unsqueeze(1).to(torch.int64),
        )
        return commit_lens, bonus_tokens, out_tokens

    def _maybe_log_accept_anomaly(
        self,
        *,
        model_worker_batch: ScheduleBatch,
        draft_input: DSparkDraftInputV2,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
        commit_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        confidence: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        prefix_lens: torch.Tensor,
        new_seq_lens: torch.Tensor,
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
        verify_cache = verify_out_cache_loc.detach().cpu().view(bs, block_size)

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
                    "input_bonus": (
                        int(input_bonus[i]) if input_bonus is not None else None
                    ),
                    "candidate0": (
                        int(candidate_row[0]) if block_size > 0 else None
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
    ) -> DSparkDraftInputV2:
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
            transfer_warmup_rounds=torch.zeros_like(seq_lens, dtype=torch.int32),
        )

    def _get_transfer_warmup_rounds(
        self, draft_input: DSparkDraftInputV2, bs: int, device: torch.device
    ) -> torch.Tensor:
        rounds = draft_input.transfer_warmup_rounds
        if rounds.numel() == bs:
            return rounds.to(device=device, dtype=torch.int32)
        return torch.zeros((bs,), dtype=torch.int32, device=device)

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
            and self.tp_rank == 0
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
            bootstrap_tokens_per_req = self._prefill_bootstrap_tokens_per_req(
                model_worker_batch.global_num_tokens
            )
            bootstrap_global_num_reqs = self._prefill_bootstrap_global_req_counts(
                model_worker_batch.global_num_tokens,
                bootstrap_tokens_per_req,
            )
            if bootstrap_global_num_reqs is not None and any(
                int(x) > 0 for x in bootstrap_global_num_reqs
            ):
                hidden_size = self._get_target_aux_hidden_size()
                hidden_dtype = (
                    logits_output.hidden_states.dtype
                    if logits_output.hidden_states is not None
                    else torch.float16
                )
                self._run_draft_bootstrap_forward(
                    batch=model_worker_batch,
                    main_hidden=torch.empty(
                        (0, hidden_size), dtype=hidden_dtype, device=device
                    ),
                    input_ids=torch.empty((0,), dtype=torch.int64, device=device),
                    positions=torch.empty((0,), dtype=torch.int64, device=device),
                    out_cache_loc=torch.empty((0,), dtype=torch.int64, device=device),
                    seq_lens=torch.empty((0,), dtype=torch.int64, device=device),
                    req_pool_indices=torch.empty(
                        (0,),
                        dtype=model_worker_batch.req_pool_indices.dtype,
                        device=device,
                    ),
                    dp_decode_global_num_tokens=bootstrap_global_num_reqs,
                    num_tokens_per_req=bootstrap_tokens_per_req,
                    use_draft_extend_v2=True,
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
        draft_seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(extend_lens)),
        )

        extend_lens_list = [int(x) for x in extend_lens]
        if not extend_lens_list:
            raise RuntimeError("DSpark prefill expected non-empty extend_lens.")
        if len(set(extend_lens_list)) == 1:
            bootstrap_global_num_reqs = self._prefill_bootstrap_global_req_counts(
                model_worker_batch.global_num_tokens,
                extend_lens_list[0],
            )
            self._run_draft_bootstrap_forward(
                batch=model_worker_batch,
                main_hidden=logits_output.hidden_states,
                input_ids=model_worker_batch.input_ids,
                positions=positions,
                out_cache_loc=model_worker_batch.out_cache_loc,
                seq_lens=draft_seq_lens,
                req_pool_indices=model_worker_batch.req_pool_indices,
                dp_decode_global_num_tokens=bootstrap_global_num_reqs,
                num_tokens_per_req=extend_lens_list[0],
                use_draft_extend_v2=True,
            )
        else:
            offset = 0
            for i, extend_len in enumerate(extend_lens_list):
                next_offset = offset + extend_len
                bootstrap_global_num_reqs = self._prefill_bootstrap_global_req_counts(
                    model_worker_batch.global_num_tokens,
                    extend_len,
                )
                self._run_draft_bootstrap_forward(
                    batch=model_worker_batch,
                    main_hidden=logits_output.hidden_states[offset:next_offset],
                    input_ids=model_worker_batch.input_ids[offset:next_offset],
                    positions=positions[offset:next_offset],
                    out_cache_loc=model_worker_batch.out_cache_loc[offset:next_offset],
                    seq_lens=draft_seq_lens[i : i + 1],
                    req_pool_indices=model_worker_batch.req_pool_indices[i : i + 1],
                    dp_decode_global_num_tokens=bootstrap_global_num_reqs,
                    num_tokens_per_req=extend_len,
                    use_draft_extend_v2=True,
                )
                offset = next_offset

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
        prefix_lens = model_worker_batch.seq_lens
        req_pool_indices = model_worker_batch.req_pool_indices

        block_ids = torch.full(
            (bs, block_size), self.noise_token_id, dtype=torch.int64, device=device
        )
        if draft_input.bonus_tokens.numel() == bs:
            block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        positions = positions_2d.reshape(-1).to(torch.int64)

        end_offset = prefix_lens + block_size
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=end_offset,
            batch_size=bs,
            draft_token_num=block_size,
            device=device,
        )
        self._materialize_disagg_prefill_hidden_to_draft_state(
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
                verify_out_cache_loc=verify_out_cache_loc,
                dp_decode_global_num_tokens=dp_decode_global_num_tokens,
            )

            candidates, confidence = self._refine_block_markov(
                block_hidden=block_hidden,
                bonus_tokens=draft_input.bonus_tokens,
                output_bs=bs,
            )

        verify_input = DSparkVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=positions,
            draft_token_num=block_size,
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
        if dp_decode_global_num_tokens is not None:
            model_worker_batch.global_num_tokens = dp_decode_global_num_tokens
            if original_global_num_tokens_for_logprob is not None:
                model_worker_batch.global_num_tokens_for_logprob = (
                    dp_decode_global_num_tokens
                )
        try:
            verify_forward_batch, _ = verify_input.prepare_for_verify(
                model_worker_batch, self.target_worker
            )
        finally:
            model_worker_batch.global_num_tokens = original_global_num_tokens
            model_worker_batch.global_num_tokens_for_logprob = (
                original_global_num_tokens_for_logprob
            )
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, block_size
        )

        new_seq_lens = None
        if bs == 0:
            bonus_tokens = torch.empty((0,), dtype=torch.int64, device=device)
            commit_lens = torch.empty((0,), dtype=torch.int32, device=device)
            out_tokens = torch.empty((0, block_size), dtype=torch.int64, device=device)
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
                logger.warning(
                    "DSPARK Triton accept/bonus failed; falling back to eager path: %s",
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
        self._maybe_log_accept_anomaly(
            model_worker_batch=model_worker_batch,
            draft_input=draft_input,
            candidates=candidates,
            target_predict=target_predict,
            commit_lens=commit_lens,
            bonus_tokens=bonus_tokens,
            confidence=confidence,
            verify_out_cache_loc=verify_out_cache_loc,
            prefix_lens=prefix_lens,
            new_seq_lens=new_seq_lens,
            ignore_mask=transfer_warmup_mask,
        )
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden = logits_output.hidden_states
        if hidden is None and bs > 0:
            raise RuntimeError(
                "DSpark verify requires target main_hidden states, but got None."
            )
        if bs > 0:
            hidden = hidden.view(bs, block_size, -1)
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            commit_mask_2d = (
                self._block_pos_offsets.unsqueeze(0)
                < commit_lens.unsqueeze(1).to(torch.int64)
            )
            commit_mask = commit_mask_2d.reshape(-1)
            self._materialize_main_hidden_to_draft_state(
                main_hidden=hidden_flat,
                cache_loc=verify_out_cache_loc[commit_mask],
                positions=positions[commit_mask],
                draft_forward_batch=draft_forward_batch,
                kv_main_hidden=hidden_flat[commit_mask],
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
            speculative_num_draft_tokens=block_size,
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
            speculative_num_draft_tokens=int(self.block_size),
            new_seq_lens=next_draft_input.new_seq_lens,
        )
