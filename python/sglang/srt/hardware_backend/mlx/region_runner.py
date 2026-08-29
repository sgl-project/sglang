"""Serve decode steps through the exported whole-model MLX region.

The region executor is a pure function over Torch-owned serving state: it
reads the KV pool through zero-copy views, returns next-token logits, and
commits the step's K/V delta back through the Torch-side Metal commit.
Torch keeps ownership of scheduling, pools, sampling, LoRA, and the whole
prefill path; this runner only replaces the decode forward.

One executor is exported lazily per decode batch size and reused for every
later step of that size (the region reads all step-to-step variability —
token ids, positions, cache slots, sequence lengths — from its tensor
arguments). A batch size whose export fails is blacklisted and served by
the eager Torch path instead.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.environ import MlxRegionStartupExport, envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.base_runner import BaseRunner

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Sizes beyond the cap fall back to the eager Torch path rather than paying
# an unbounded set of exports.
_MAX_REGION_BATCH_SIZE = 16

# Prefill token counts vary per request, so extend executors are exported per
# padded bucket instead of per exact length. The x1.3-1.5 ladder bounds pad
# waste well below the region's kernel win.
_PREFILL_TOKEN_BUCKETS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)

# Where a padded prefill token's K/V goes. Slot 0 is the pool's reserved
# padding row: both allocators build their free list as ``arange(1, size + 1)``
# under the comment "The padded slot 0 is used for writing dummy outputs from
# padded tokens" (``mem_cache/allocator/token.py``, ``allocator/paged.py``),
# and the pools carry a +1 row at index 0 for it. Any other index -- including
# ``token_to_kv_pool.size``, which is in bounds because of that extra row --
# is allocatable, so pad rows would overwrite a live request's K/V with no
# error once the pool neared exhaustion.
_PAD_SINK_SLOT = 0

# Decode shapes exported at startup under SGLANG_MLX_REGION_STARTUP_EXPORT.
# Mirrors the spirit of the CUDA-graph capture ladder; sizes off the ladder
# still export lazily at their first batch.
_STARTUP_DECODE_BATCH_SIZES = (1, 2, 4, 8, 16)


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.clear_cache()


def _nontrivial_logits_reason(model: Any) -> Optional[str]:
    """Why the exported hidden @ lm_head.T shortcut would be wrong, or None.

    Model classes vary in shape, so this probes attributes instead of a type.
    """
    lm_head = getattr(model, "lm_head", None)
    processor = getattr(model, "logits_processor", None)
    if lm_head is None or processor is None:
        return "model exposes no lm_head/logits_processor"
    if getattr(processor, "logit_scale", None) is not None:
        return "LogitsProcessor applies logit_scale"
    if getattr(processor, "final_logit_softcapping", None) is not None:
        return "LogitsProcessor applies final-logit softcapping"
    return None


def _kernel_contract_reject_reason(model_runner: Any) -> Optional[str]:
    """Why the region's Metal kernels cannot serve this model, or None.

    The exported executor templates one uniform (query heads, KV heads,
    head dim) attention spec from the KV pool, bakes ``head_dim ** -0.5`` as
    the attention scale, and commits stacked per-layer K/V deltas in
    bfloat16. A model outside that contract must be rejected before any
    batch reaches device work: the kernel entry points raise at dispatch,
    which would crash serving instead of falling back.
    """
    from sglang.kernels.ops.attention.mlx_radix_attention import (
        deferred_attention_reject_reason,
    )
    from sglang.srt.hardware_backend.mlx.export_validation import (
        ensure_model_layers,
    )
    from sglang.srt.layers.radix_attention import AttentionType, RadixAttention

    try:
        ensure_model_layers(model_runner)
    except RuntimeError as error:
        # The topology resolver raises one explicit, actionable error naming
        # the model class; surface it as the pre-launch rejection reason.
        return str(error)
    if not model_runner.attention_layers:
        return "no attention layers discovered in the decoder layer stack"
    kv_geometries = set()
    for layer_id, layer in enumerate(model_runner.attention_layers):
        if not isinstance(layer, RadixAttention):
            return (
                f"layer {layer_id} is not a plain RadixAttention layer; "
                "hybrid attention stacks are not implemented by the region"
            )
        if layer.attn_type != AttentionType.DECODER or layer.is_cross_attention:
            return f"layer {layer_id} is not decoder self-attention"
        if layer.sliding_window_size > 0:
            return f"layer {layer_id} uses sliding-window attention"
        if layer.logit_cap:
            return f"layer {layer_id} applies attention logit capping"
        if layer.qk_head_dim != layer.head_dim or layer.v_head_dim != layer.head_dim:
            return (
                f"layer {layer_id} uses split QK/V head dims "
                f"({layer.qk_head_dim}/{layer.v_head_dim}); the kernels "
                "assume one head_dim"
            )
        geometry_reason = deferred_attention_reject_reason(
            num_q_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_dim=layer.head_dim,
        )
        if geometry_reason is not None:
            return f"attention kernel cannot serve layer {layer_id}: {geometry_reason}"
        if not math.isclose(layer.scaling, layer.head_dim**-0.5, rel_tol=1e-6):
            return (
                f"layer {layer_id} scales attention by {layer.scaling} but "
                f"the kernels bake head_dim ** -0.5 "
                f"({layer.head_dim**-0.5:.6g})"
            )
        kv_geometries.add((layer.tp_k_head_num, layer.head_dim))
    if len(kv_geometries) > 1:
        return (
            "KV geometry differs across layers; the stacked KV-delta commit "
            f"requires one shape, found {sorted(kv_geometries)}"
        )
    pool = model_runner.token_to_kv_pool
    k_cache, _ = pool.get_kv_buffer(pool.start_layer)
    if k_cache.dtype != torch.bfloat16 or model_runner.dtype != torch.bfloat16:
        return (
            "region kernels require bfloat16 activations and KV pools, found "
            f"model dtype {model_runner.dtype} and KV dtype {k_cache.dtype}"
        )
    kv_geometry = next(iter(kv_geometries))
    if tuple(k_cache.shape[1:]) != kv_geometry:
        return (
            f"KV pool row shape {tuple(k_cache.shape[1:])} does not match "
            f"the attention layers' (KV heads, head_dim) {kv_geometry}"
        )
    return None


class MlxRegionRunner(BaseRunner):
    """Decode-only graph runner backed by the exported MLX region."""

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        from sglang.srt.hardware_backend.mps.runtime import validate_mps_runtime

        validate_mps_runtime()
        self._lora_enabled = bool(model_runner.server_args.enable_lora)
        self._executors: dict[int, Any] = {}
        self._failed_batch_sizes: set[int] = set()
        self._state_token: Optional[tuple] = None
        self._constants_checked = False
        # The wrapper computes hidden @ lm_head.T directly; any model whose
        # LogitsProcessor does more than that would silently diverge.
        self._model_reject_reason = _nontrivial_logits_reason(model_runner.model)
        if (
            self._model_reject_reason is None
            and (model_runner.sliding_window_size or 0) > 0
        ):
            # The region attention kernels attend the full context; a
            # sliding-window model would be silently wrong, not slow.
            self._model_reject_reason = (
                "sliding-window attention is not implemented by the region"
            )
        if self._model_reject_reason is None:
            # The Metal kernels raise at dispatch on geometry or dtype they
            # do not support; reject at startup so no batch reaches device
            # work with the contract unsatisfied.
            self._model_reject_reason = _kernel_contract_reject_reason(model_runner)
        if self._model_reject_reason is not None:
            logger.warning(
                "MLX region disabled for this model: %s. Decode serves on the "
                "eager Torch path.",
                self._model_reject_reason,
            )
        self.startup_export_seconds = 0.0
        if self._model_reject_reason is None:
            self._export_at_startup()

    def _startup_keys(self) -> list[tuple]:
        level = envs.SGLANG_MLX_REGION_STARTUP_EXPORT.get()
        if level <= MlxRegionStartupExport.OFF:
            return []
        keys: list[tuple] = [
            ("decode", batch_size) for batch_size in _STARTUP_DECODE_BATCH_SIZES
        ]
        if level >= MlxRegionStartupExport.ALL:
            keys.extend(("extend", bucket) for bucket in _PREFILL_TOKEN_BUCKETS)
        return keys

    def _export_at_startup(self) -> None:
        """Export (and warm) the configured shape ladder before serving.

        Same contract as CUDA-graph capture at startup: each shape pays its
        export once here instead of on its first request. Shapes whose
        export fails are blacklisted exactly as they would be at serve time.
        One warm execution per shape also triggers the MLX graph compile,
        so the first real batch is a pure cache hit.
        """
        keys = self._startup_keys()
        if not keys:
            return
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        started = time.perf_counter()
        exported = 0
        # Serving installs the forward context before dispatch reaches this
        # runner; the export trace and the warm execution read the attention
        # backend through it, so install the same context here.
        context = ForwardContext(attn_backend=self.model_runner.attn_backend)
        for key in keys:
            if self._model_reject_reason is not None:
                break
            batch = self._synthetic_batch(key)
            with forward_context(context):
                if self._ensure_executor(batch, key) is None:
                    continue
                try:
                    with torch.inference_mode():
                        self.execute(batch)
                except Exception:
                    logger.exception(
                        "MLX region warm-up execution failed for %s; serving "
                        "this shape on the eager Torch path.",
                        key,
                    )
                    self._executors.pop(key, None)
                    self._failed_batch_sizes.add(key)
                    continue
            exported += 1
            # Each export leaves transient buffers in the MLX cache; release
            # them so the ladder's peak footprint stays that of one shape.
            _clear_mlx_cache()
        self.startup_export_seconds = time.perf_counter() - started
        logger.info(
            "MLX region: exported %d/%d shapes at startup in %.1f s.",
            exported,
            len(keys),
            self.startup_export_seconds,
        )

    def _synthetic_batch(self, key: tuple) -> ForwardBatch:
        """A dummy batch of the key's shape, using the pool's reserved rows.

        Row 0 of req_to_token and slot 0 of the KV pool are the reserved
        padding entries (the same ones CUDA-graph capture points dummy
        batches at), so the export trace and the warm execution read and
        write nothing a request owns. Field dtypes match the serving path
        exactly: the exported executor binds them into its signature.
        """
        device = self.model_runner.device
        mode, size = key
        if mode == "decode":
            zeros = torch.zeros(size, dtype=torch.int64, device=device)
            return ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=size,
                input_ids=zeros,
                positions=zeros.clone(),
                req_pool_indices=zeros.clone(),
                seq_lens=torch.ones(size, dtype=torch.int64, device=device),
                out_cache_loc=zeros.clone(),
                seq_lens_sum=size,
                num_token_non_padded_cpu=size,
            )
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.zeros(size, dtype=torch.int64, device=device),
            positions=torch.arange(size, dtype=torch.int64, device=device),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device=device),
            seq_lens=torch.full((1,), size, dtype=torch.int64, device=device),
            out_cache_loc=torch.zeros(size, dtype=torch.int64, device=device),
            seq_lens_sum=size,
            extend_num_tokens=size,
            extend_seq_lens=torch.full((1,), size, dtype=torch.int32, device=device),
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
            extend_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
            num_token_non_padded_cpu=size,
        )

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        if self._model_reject_reason is not None:
            return False
        if forward_batch.spec_info is not None:
            return False
        if forward_batch.encoder_lens is not None:
            return False
        if forward_batch.capture_hidden_mode not in (None, CaptureHiddenMode.NULL):
            return False
        if self._lora_enabled or self.model_runner.hisparse_coordinator is not None:
            return False
        key = self._executor_key(forward_batch)
        if key is None or key in self._failed_batch_sizes:
            return False
        return self._ensure_executor(forward_batch, key) is not None

    def _executor_key(self, forward_batch: ForwardBatch) -> Optional[tuple]:
        """The executor cache key, or None when the batch shape is unservable."""
        mode = forward_batch.forward_mode
        if mode.is_decode():
            if forward_batch.batch_size > _MAX_REGION_BATCH_SIZE:
                return None
            return ("decode", forward_batch.batch_size)
        # Strictly plain EXTEND: is_extend() also admits MIXED / TARGET_VERIFY /
        # DLLM variants whose semantics the exported graph does not carry.
        if mode != ForwardMode.EXTEND:
            return None
        if forward_batch.batch_size != 1:
            return None
        if forward_batch.return_logprob:
            # Prompt logprobs need the full LogitsProcessor machinery.
            return None
        if (
            forward_batch.input_embeds is not None
            or forward_batch.replace_embeds is not None
        ):
            return None
        num_tokens = forward_batch.input_ids.shape[0]
        for bucket in _PREFILL_TOKEN_BUCKETS:
            if num_tokens <= bucket:
                return ("extend", bucket)
        return None

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Any = None,
        **kwargs: Any,
    ) -> ForwardBatch:
        return forward_batch

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Any = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        from sglang.srt.hardware_backend.mlx.export_validation import (
            serving_forward_args,
        )

        key = self._executor_key(forward_batch)
        executor = self._executors[key]
        if key[0] == "extend":
            num_tokens = forward_batch.input_ids.shape[0]
            padded = self._pad_extend_batch(forward_batch, key[1])
            logits = executor.execute(*serving_forward_args(padded))
            # The wrapper emits logits for every packed token; serving wants
            # the last real token's row (pad rows carry garbage by design).
            return LogitsProcessorOutput(
                next_token_logits=logits[num_tokens - 1 : num_tokens]
            )
        logits = executor.execute(*serving_forward_args(forward_batch))
        return LogitsProcessorOutput(next_token_logits=logits)

    def _pad_extend_batch(
        self, forward_batch: ForwardBatch, bucket: int
    ) -> ForwardBatch:
        """Pad the token-dimension tensors to the executor's bucket shape.

        Pad rows are causal-safe (appended after every real token), their
        RoPE positions repeat the last real position, and their K/V rows are
        committed to the pool's reserved padding slot ``_PAD_SINK_SLOT``.
        """
        num_tokens = forward_batch.input_ids.shape[0]
        pad = bucket - num_tokens
        if pad == 0:
            return forward_batch
        return dataclasses.replace(
            forward_batch,
            input_ids=torch.cat(
                [
                    forward_batch.input_ids,
                    forward_batch.input_ids.new_zeros(pad),
                ]
            ),
            positions=torch.cat(
                [
                    forward_batch.positions,
                    forward_batch.positions[-1:].expand(pad),
                ]
            ),
            out_cache_loc=torch.cat(
                [
                    forward_batch.out_cache_loc,
                    forward_batch.out_cache_loc.new_full((pad,), _PAD_SINK_SLOT),
                ]
            ),
        )

    def _state_identity(self) -> tuple:
        """Storage identity of everything the exported views alias.

        Covers KV-pool reallocation and weight *replacement* (new storage).
        In-place weight updates keep the same storage, which the views alias,
        so they stay correct without a re-export.
        """
        k_cache, _ = self.model_runner.token_to_kv_pool.get_kv_buffer(
            self.model_runner.token_to_kv_pool.start_layer
        )
        model = self.model_runner.model
        first_param = next(model.parameters())
        return (
            k_cache.data_ptr(),
            model.lm_head.weight.data_ptr(),
            first_param.data_ptr(),
        )

    def _ensure_executor(
        self, forward_batch: ForwardBatch, key: tuple
    ) -> Optional[Any]:
        # Reallocated pools or replaced weights invalidate the zero-copy
        # views bound at export.
        state_token = self._state_identity()
        if state_token != self._state_token:
            if self._executors:
                logger.info(
                    "MLX region: KV pool or weight storage changed; "
                    "re-exporting regions."
                )
            self._executors.clear()
            self._failed_batch_sizes.clear()
            self._state_token = state_token

        executor = self._executors.get(key)
        if executor is not None:
            return executor

        from sglang.srt.compilation.torch_compile_decoration import _to_torch
        from sglang.srt.hardware_backend.mlx.export_validation import (
            build_serving_mlx_executor,
            serving_export_context,
        )

        # Extend executors are exported at their padded bucket shape so one
        # export serves every prompt length the bucket covers.
        export_batch = (
            self._pad_extend_batch(forward_batch, key[1])
            if key[0] == "extend"
            else forward_batch
        )
        start = time.perf_counter()
        try:
            with serving_export_context(self.model_runner, export_batch):
                region = build_serving_mlx_executor(self.model_runner, export_batch)
        except Exception:
            logger.exception(
                "MLX region export failed for %s; serving this shape on the "
                "eager Torch path.",
                key,
            )
            self._failed_batch_sizes.add(key)
            return None
        finally:
            # Export switches fused ops to their compile-safe forwards;
            # restore them so the eager Torch path keeps its fast kernels.
            _to_torch(
                self.model_runner.model,
                reverse=True,
                num_tokens=export_batch.input_ids.shape[0],
            )
        logger.info(
            "MLX region exported for %s in %.1f s.",
            key,
            time.perf_counter() - start,
        )
        if not self._constants_checked and not self._graph_ignores_baked_constants(
            region, export_batch
        ):
            self._model_reject_reason = (
                "exported graph depends on batch fields the wrapper bakes as "
                "constants; executor reuse across steps would be unsound"
            )
            logger.warning(
                "MLX region disabled for this model: %s. Decode serves on "
                "the eager Torch path.",
                self._model_reject_reason,
            )
            self._executors.clear()
            return None
        if not self._constants_checked:
            logger.info(
                "MLX region: exported graph verified independent of baked "
                "batch constants; executor reuse across steps is sound."
            )
            self._constants_checked = True
        self._executors[key] = region
        return region

    def _graph_ignores_baked_constants(
        self, region: Any, forward_batch: ForwardBatch
    ) -> bool:
        """Prove the export is a function of tensor arguments only.

        The wrapper bakes ``seq_lens_sum`` and ``num_token_non_padded_cpu`` as
        Python constants, but executors are reused across steps where those
        values change. Re-export once with perturbed constants and require an
        identical op sequence; run once per model.
        """
        import dataclasses

        from sglang.srt.hardware_backend.mlx.export_validation import (
            serving_export_context,
            serving_graph_signature,
        )

        baseline = tuple(
            f"{node.op}:{node.target}"
            for node in region.exported_program.graph_module.graph.nodes
        )
        perturbed_batch = dataclasses.replace(
            forward_batch,
            seq_lens_sum=forward_batch.seq_lens_sum + 1,
            num_token_non_padded_cpu=(
                None
                if forward_batch.num_token_non_padded_cpu is None
                else forward_batch.num_token_non_padded_cpu + 1
            ),
        )
        try:
            with serving_export_context(self.model_runner, perturbed_batch):
                perturbed = serving_graph_signature(self.model_runner, perturbed_batch)
        except Exception:
            logger.exception(
                "MLX region: constant-independence re-export failed; "
                "treating the model as dependent on baked constants."
            )
            return False
        return baseline == perturbed
