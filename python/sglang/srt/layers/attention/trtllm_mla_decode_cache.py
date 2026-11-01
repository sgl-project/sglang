"""Skip flashinfer's per-call MLA decode dispatch work in trtllm_mla eager decode.

``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`` builds its runner
objects and tuning config and asks ``AutoTuner.choose_one`` for a tactic on
every call. CUDA-graph replay never runs that host code, but eager decode pays
it once per layer per step.

``cached_trtllm_mla_decode`` keeps only the ``(runner_kind, tactic)`` decision
chosen across FlashInfer's candidate runners (``trtllm-gen`` and ``cute-dsl``)
for a workload shape and builds the selected runner on every call from the
caller's tensors and exact ``max_seq_len``:

- Outputs are bitwise identical to the public API, both untuned and after
  ``flashinfer.autotune(True)``.
- No tensor is cached, so the KV pool and workspace can be freed or re-created
  at any time (memory saver, weight sync).
- Entries hold no per-layer state, so all layers share the same entry within a
  FlashInfer seq-len / 128-token K-tile bucket.

The fast path needs the runner classes in ``flashinfer.mla._core`` (flashinfer
<= 0.7.2). Calls that request the LSE, flashinfer builds without those classes,
and builds whose dispatcher caches the decision itself
(``AutoTuner.selection_cache_token``) use the public API unchanged.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Keys hold no per-layer state, so the cache grows with the number of distinct
# (batch size, max_seq_len bucket) pairs only.
_TACTIC_CACHE_MAX_ENTRIES = 4096


class _LruCache(OrderedDict):
    def __init__(self, max_entries: int):
        super().__init__()
        self.max_entries = max_entries

    def lookup(self, key):
        value = self.get(key)
        if value is not None:
            self.move_to_end(key)
        return value

    def insert(self, key, value) -> None:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)


_TACTIC_CACHE = _LruCache(_TACTIC_CACHE_MAX_ENTRIES)
_DEVICE_INFO_CACHE: dict = {}
_CUTE_DSL_SUPPORT_CACHE: dict = {}
_FI_RUNNER_API = (
    None  # None: not probed yet; False: use the public API; tuple: symbols.
)


def _device_info(device: torch.device) -> tuple[int, int]:
    info = _DEVICE_INFO_CACHE.get(device)
    if info is None:
        props = torch.cuda.get_device_properties(device)
        info = (props.major, props.multi_processor_count)
        _DEVICE_INFO_CACHE[device] = info
    return info


def _flashinfer_runner_api():
    """flashinfer's MLA decode runner classes and helpers, or None.

    None means the public API should be used: either the classes are gone, or
    the dispatcher already caches its decision (newer flashinfer).
    """
    global _FI_RUNNER_API
    if _FI_RUNNER_API is None:
        try:
            from flashinfer.autotuner import AutoTuner
            from flashinfer.mla._core import (
                CuteDslMlaDecodeRunner,
                TrtllmGenMlaDecodeRunner,
                _build_mla_decode_tuning_config,
                _cute_dsl_incompatibility_reason,
                _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
                _round_to_seq_len_bucket,
            )
        except ImportError as e:
            logger.info(
                "flashinfer MLA decode runner classes are unavailable (%s); "
                "trtllm_mla decode uses the public flashinfer API.",
                e,
            )
            _FI_RUNNER_API = False
        else:
            if hasattr(AutoTuner, "selection_cache_token"):
                _FI_RUNNER_API = False  # flashinfer caches the dispatch decision itself
            else:
                _FI_RUNNER_API = (
                    AutoTuner,
                    CuteDslMlaDecodeRunner,
                    TrtllmGenMlaDecodeRunner,
                    _build_mla_decode_tuning_config,
                    _cute_dsl_incompatibility_reason,
                    _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
                    _round_to_seq_len_bucket,
                )
    return _FI_RUNNER_API or None


def _cute_dsl_reason(
    incompat_fn,
    query_4d: torch.Tensor,
    out_dtype: torch.dtype,
    bmm1_scale,
    bmm2_scale,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    page_size: int,
) -> Optional[str]:
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        return incompat_fn(
            query_4d,
            out_dtype,
            bmm1_scale,
            bmm2_scale,
            None,
            0,
            None,
            True,
            qk_rope_head_dim,
            kv_lora_rank,
            page_size,
            True,
            use_fp16_softmax=False,
            cute_dsl_impl="auto",
            enable_dcp=False,
            cp_world=1,
        )
    key = (
        query_4d.device,
        query_4d.dtype,
        out_dtype,
        page_size,
        query_4d.shape[1],
        query_4d.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
    )
    if key not in _CUTE_DSL_SUPPORT_CACHE:
        _CUTE_DSL_SUPPORT_CACHE[key] = incompat_fn(
            query_4d,
            out_dtype,
            bmm1_scale,
            bmm2_scale,
            None,
            0,
            None,
            True,
            qk_rope_head_dim,
            kv_lora_rank,
            page_size,
            True,
            use_fp16_softmax=False,
            cute_dsl_impl="auto",
            enable_dcp=False,
            cp_world=1,
        )
    return _CUTE_DSL_SUPPORT_CACHE[key]


def _tuner_token(tuner) -> Optional[tuple]:
    """None while tuning; otherwise a value that changes when tuner state changes."""
    if tuner.is_tuning_mode or tuner._active_tuning_contexts:
        return None
    return (
        id(tuner),
        tuner._dirty_seq,
        len(tuner.profiling_cache),
        len(tuner._file_configs),
    )


def _public_decode(
    *,
    query,
    kv_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    bmm1_scale,
    bmm2_scale,
    qk_nope_head_dim,
    kv_lora_rank,
    qk_rope_head_dim,
    workspace_buffer,
    multi_ctas_kv_counter_buffer,
    out,
    return_lse,
    enable_pdl,
    backend,
):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    # Same keyword arguments as TRTLLMMLABackend's uncached call.
    extra_kwargs = {"backend": backend} if backend != "trtllm-gen" else {}
    if backend == "trtllm-gen":
        extra_kwargs["multi_ctas_kv_counter_buffer"] = multi_ctas_kv_counter_buffer
    return trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        enable_pdl=enable_pdl,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        return_lse=return_lse,
        **extra_kwargs,
    )


def cached_trtllm_mla_decode(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    bmm1_scale: float,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    workspace_buffer: torch.Tensor,
    bmm2_scale: float = 1.0,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    enable_pdl: Optional[bool] = True,
    backend: str = "trtllm-gen",
):
    """Dense MLA decode that reuses flashinfer's autotuner tactic per shape.

    Returns what ``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla``
    returns for the same arguments, bit for bit. ``backend`` is
    ``"trtllm-gen"`` or ``"cute-dsl"``.
    """
    if backend not in ("trtllm-gen", "cute-dsl"):
        raise ValueError(f"backend must be 'trtllm-gen' or 'cute-dsl', got {backend!r}")
    if (
        return_lse
        or seq_lens is None
        or query.ndim != 4
        or kv_cache.ndim not in (3, 4)
        or block_tables.ndim != 2
        or isinstance(bmm1_scale, torch.Tensor)
        or isinstance(bmm2_scale, torch.Tensor)
    ):
        api = None
    else:
        api = _flashinfer_runner_api()
    if api is None:
        return _public_decode(
            query=query,
            kv_cache=kv_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            workspace_buffer=workspace_buffer,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
            out=out,
            return_lse=return_lse,
            enable_pdl=enable_pdl,
            backend=backend,
        )
    (
        AutoTuner,
        CuteDslMlaDecodeRunner,
        TrtllmGenMlaDecodeRunner,
        build_tuning_config,
        cute_dsl_incompat_reason,
        resolve_counter_buffer,
        round_to_seq_len_bucket,
    ) = api

    dev = query.device
    cc_major, sm_count = _device_info(dev)
    kv_4d = kv_cache.unsqueeze(1) if kv_cache.ndim == 3 else kv_cache
    page_size = kv_4d.shape[2]
    B, q_len, num_heads, qk_head_dim = query.shape
    max_seq_len = int(max_seq_len)
    expected_qk_dim = kv_lora_rank + qk_rope_head_dim

    # Match FlashInfer's SM10x dispatch and _check_trtllm_gen_mla_shape checks;
    # non-SM10x (e.g. SM120 XQA) and invalid/boundary shapes delegate to _public_decode.
    if (
        cc_major != 10
        or B <= 0
        or q_len <= 0
        or max_seq_len <= 0
        or qk_rope_head_dim != 64
        or (kv_lora_rank != 512 and kv_lora_rank != 256)
        or qk_head_dim != expected_qk_dim
        or kv_4d.shape[1] != 1
        or kv_4d.shape[3] != expected_qk_dim
        or block_tables.shape[0] != B
        or page_size not in (32, 64)
        or (block_tables.shape[-1] * page_size) % 128 != 0
        or (
            out is not None
            and (
                out.shape != (B, q_len, num_heads, kv_lora_rank)
                or out.dtype != torch.bfloat16
                or out.device != dev
            )
        )
    ):
        return _public_decode(
            query=query,
            kv_cache=kv_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            workspace_buffer=workspace_buffer,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
            out=out,
            return_lse=return_lse,
            enable_pdl=enable_pdl,
            backend=backend,
        )

    if enable_pdl is None:
        enable_pdl = True
    if out is None:
        out = torch.empty(
            (B, q_len, num_heads, kv_lora_rank),
            dtype=torch.bfloat16,
            device=dev,
        )

    cute_reason = _cute_dsl_reason(
        cute_dsl_incompat_reason,
        query,
        out.dtype,
        bmm1_scale,
        bmm2_scale,
        qk_rope_head_dim,
        kv_lora_rank,
        page_size,
    )
    if backend == "cute-dsl":
        if cute_reason is not None:
            raise ValueError(cute_reason)
        if multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported when a "
                "trtllm-gen runner is selected"
            )
        runner_names = ["cute-dsl"]
    else:
        # TRTLLMMLABackend omits `backend` when self.backend == "trtllm-gen",
        # so flashinfer executes backend="auto" across both candidate runners.
        trt_ok = not (64 < num_heads < 128) and page_size in (32, 64)
        runner_names = []
        if trt_ok:
            runner_names.append("trtllm-gen")
        if cute_reason is None:
            runner_names.append("cute-dsl")
        if not runner_names or (
            multi_ctas_kv_counter_buffer is not None
            and "trtllm-gen" not in runner_names
        ):
            return _public_decode(
                query=query,
                kv_cache=kv_cache,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                workspace_buffer=workspace_buffer,
                multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
                out=out,
                return_lse=return_lse,
                enable_pdl=enable_pdl,
                backend=backend,
            )

    def _make_runner(kind: str):
        if kind == "cute-dsl":
            return CuteDslMlaDecodeRunner(
                kv_cache=kv_4d,
                workspace_buffer=workspace_buffer,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                softmax_scale=bmm1_scale,
                output_scale=bmm2_scale,
                out_dtype=out.dtype,
                enable_pdl=enable_pdl,
                is_var_seq=True,
                uses_shared_paged_kv_idx=True,
                lse=None,
                return_lse=False,
                sinks=None,
                cute_dsl_impl="auto",
                enable_dcp=False,
                cp_world=1,
                cp_rank=0,
            )
        return TrtllmGenMlaDecodeRunner(
            kv_cache=kv_4d,
            workspace_buffer=workspace_buffer,
            sm_count=sm_count,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            sparse_mla_top_k=0,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=None,
            skip_softmax_threshold_scale_factor=None,
            enable_pdl=enable_pdl,
            is_var_seq=True,
            uses_shared_paged_kv_idx=True,
            return_lse=False,
            lse=None,
            use_fp16_softmax=False,
        )

    inputs = [query, block_tables, seq_lens, out]

    tuner = AutoTuner.get()
    token = _tuner_token(tuner)
    # flashinfer's autotune key buckets max_seq_len (power of two; 128-token K
    # tiles for the CuTe DSL workspace), so the tactic is shared across a bucket.
    cache_key = (
        backend,
        tuple(runner_names),
        dev,
        kv_4d.dtype,
        tuple(kv_4d.shape[1:]),
        query.dtype,
        out.dtype,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        block_tables.shape[-1],
        B,
        q_len,
        num_heads,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        round_to_seq_len_bucket(max_seq_len),
        (max_seq_len + 127) // 128 if "cute-dsl" in runner_names else None,
        float(bmm1_scale),
        float(bmm2_scale),
        bool(enable_pdl),
    )
    entry = _TACTIC_CACHE.lookup(cache_key) if token is not None else None
    if entry is not None and entry[0] == token:
        _, chosen_kind, tactic = entry
        chosen_runner = _make_runner(chosen_kind)
    else:
        runners = [_make_runner(kind) for kind in runner_names]
        tuning_config = build_tuning_config(
            kv_cache=kv_4d,
            block_tables=block_tables,
            workspace_buffer=workspace_buffer,
            runner_names=runner_names,
            q_len=q_len,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            max_seq_len=max_seq_len,
            device=dev,
            has_sparse_mla_top_k_lens=False,
            cute_dsl_impl="auto",
            sinks=None,
            enable_dcp=False,
            cp_world=1,
            cp_rank=0,
        )
        chosen_runner, tactic = tuner.choose_one(
            "trtllm_batch_decode_mla",
            runners,
            tuning_config,
            inputs,
        )
        chosen_kind = (
            "cute-dsl"
            if isinstance(chosen_runner, CuteDslMlaDecodeRunner)
            else "trtllm-gen"
        )
        del runners
        if token is not None and _tuner_token(tuner) == token:
            _TACTIC_CACHE.insert(cache_key, (token, chosen_kind, tactic))

    if chosen_kind == "trtllm-gen":
        if multi_ctas_kv_counter_buffer is not None:
            multi_ctas_kv_counter_buffer = resolve_counter_buffer(
                multi_ctas_kv_counter_buffer, B, num_heads, sm_count, dev
            )
        chosen_runner(
            inputs=inputs,
            tactic=tactic,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
    else:
        chosen_runner(inputs=inputs, tactic=tactic)
    return out
