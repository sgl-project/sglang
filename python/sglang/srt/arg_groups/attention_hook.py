# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the attention backends."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    _attention_backend_default,
    _attention_backend_dual_chunk,
    _attention_backend_fa3_fp8_fallback,
    _attention_backend_platform_fallbacks,
    _cutedsl_prefill_backend_fill,
    _deterministic_allreduce_fusion_disable,
    _deterministic_attention_backend,
    _deterministic_sampling_backend,
    _fa4_page_constraint,
    _intel_xpu_page_constraint,
    _mla_backend_page_constraints,
    _mla_kv_cache_dtype_checks,
    attention_backends_of,
    declare_resolution,
    mamba_extra_buffer_of,
    model_config_of,
    resolved_view,
    resolving_view,
    run_post_process_pass,
    use_mla_backend,
)
from sglang.srt.connector import ConnectorType
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import (
    parse_connector_type,
)

logger = logging.getLogger(__name__)


def handle_attention_backend_compatibility(server_args: Any):

    cfg = resolving_view(server_args)
    model_config = model_config_of(server_args)

    # The attention_backend write clusters of this handler moved to the
    # resolution pipeline (arg_groups/overrides.py), each invoked below at
    # its legacy slot; the interleaved non-attention adjustments stay.

    # Split-backend override + default fill.
    run_post_process_pass(server_args, _attention_backend_default)

    # Torch native and flex attention backends
    attention_backend = resolved_view(server_args).attention_backend
    if attention_backend == "torch_native":
        logger.warning(
            "Cuda graph is disabled because of using torch native attention backend"
        )
        declare_resolution(
            server_args,
            "_handle_attention_backend_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args,
            "_handle_attention_backend_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )

    if attention_backend == "flex_attention":
        logger.warning(
            "Cuda graph is disabled because of using torch Flex Attention backend"
        )
        declare_resolution(
            server_args,
            "_handle_attention_backend_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args,
            "_handle_attention_backend_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )
        assert (
            cfg.speculative_algorithm is None
        ), "Speculative decoding is currently not supported with Flex Attention backend"

    # Whisper's encoder token padding conflicts with prefix caching.
    # Only disable for Whisper; other encoder-decoder models (e.g., mllama) use radix cache.
    if (
        model_config.is_encoder_decoder
        and not cfg.disable_radix_cache
        and "WhisperForConditionalGeneration"
        in (model_config.hf_config.architectures or [])
    ):
        logger.info("Radix cache is disabled for Whisper")
        declare_resolution(
            server_args,
            "_handle_attention_backend_compatibility",
            disable_radix_cache=True,
        )

    # Major NVIDIA platforms backends: the page-size snaps of this family
    # moved to the resolution pipeline (arg_groups/overrides.py:
    # _mla_backend_page_constraints); the raises and the cutedsl prefill
    # fallback stay below.
    run_post_process_pass(server_args, _mla_backend_page_constraints)

    # The TRT-LLM / tokenspeed MLA kv-dtype validations moved to the
    # resolution pipeline (arg_groups/overrides.py:
    # _mla_kv_cache_dtype_checks), invoked here at their legacy slot.

    run_post_process_pass(server_args, _mla_kv_cache_dtype_checks)

    # The CuteDSL MLA validation + prefill fill moved to the resolution
    # pipeline (arg_groups/overrides.py: _cutedsl_prefill_backend_fill),
    # invoked here at its legacy slot.

    run_post_process_pass(server_args, _cutedsl_prefill_backend_fill)

    prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
    if "trtllm_mha" in (prefill_backend, decode_backend):
        if prefill_backend == "trtllm_mha" and not (
            get_platform().is_sm90 or get_platform().is_sm100 or get_platform().is_sm120
        ):
            raise ValueError(
                "TRTLLM MHA backend for prefill requires Hopper (SM90), Blackwell (SM100), or SM120 GPUs. "
                "Please use a different prefill backend."
            )
        if (
            prefill_backend == "trtllm_mha"
            and get_platform().is_sm120
            and (
                cfg.kv_cache_dtype == "fp8_e4m3"
                or (
                    envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get() or 0.0
                )
                > 0
            )
        ):
            raise ValueError(
                "TRTLLM FMHAv2 prefill on SM120 does not support "
                "fp8_e4m3 KV cache or skip-softmax."
            )
        if decode_backend == "trtllm_mha" and not (
            get_platform().is_sm90 or get_platform().is_sm100 or get_platform().is_sm120
        ):
            raise ValueError(
                "TRTLLM MHA backend for decode is only supported on Hopper (SM90), Blackwell (SM100) and (SM120) GPUs. Please use a different decode backend."
            )
        if (
            prefill_backend == "trtllm_mha"
            and not get_platform().is_sm100
            and (cfg.enable_prefill_context_parallel or cfg.attn_cp_size > 1)
        ):
            raise ValueError(
                "Prefill context parallelism with the TRTLLM MHA prefill backend "
                "requires SM100 (trtllm-gen context kernel): the SM90/SM120 "
                "fmha_v2 prefill path does not implement CP shard masking."
            )

    run_post_process_pass(server_args, _attention_backend_fa3_fp8_fallback)

    run_post_process_pass(server_args, _fa4_page_constraint)

    # AMD platforms backends
    if resolved_view(server_args).attention_backend == "aiter":
        if model_config.context_len > 8192:
            # The 0.85 covers the extra non-static workspace aiter reserves for
            # long contexts, but it is a heuristic for the auto-derived default
            # only. Shrinking a value the user picked can push the static budget
            # below the model-weight footprint on a nearly full GPU and break
            # KV-cache allocation outright, so an explicit value is honored.
            if (getattr(server_args, "_raw_input", None) or {}).get(
                "mem_fraction_static"
            ) is not None:
                logger.warning(
                    "attention_backend=aiter with context_len=%d (>8192) "
                    "normally scales mem_fraction_static by 0.85, but "
                    "mem_fraction_static=%.3f was set explicitly and will be "
                    "used as-is. Ensure enough non-static memory is left for "
                    "attention workspace and CUDA graphs.",
                    model_config.context_len,
                    cfg.mem_fraction_static,
                )
            else:
                declare_resolution(
                    server_args,
                    "_handle_attention_backend_compatibility",
                    mem_fraction_static=cfg.mem_fraction_static * 0.85,
                )

    # Other platforms backends
    run_post_process_pass(server_args, _attention_backend_platform_fallbacks)

    prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
    if use_mla_backend(server_args) and prefill_backend == "intel_xpu":
        raise ValueError(
            "intel_xpu backend is only supported on decode for MLA models, please set --decode-attention-backend to intel_xpu and do not set --attention-backend or --prefill-attention-backend to intel_xpu for prefill instead use triton."
        )

    run_post_process_pass(server_args, _intel_xpu_page_constraint)

    # Dual chunk flash attention backend
    run_post_process_pass(server_args, _attention_backend_dual_chunk)
    if resolved_view(server_args).attention_backend == "dual_chunk_flash_attn":
        logger.warning(
            "Mixed chunk and radix cache are disabled when using dual-chunk flash attention backend"
        )
        declare_resolution(
            server_args,
            "_handle_attention_backend_compatibility",
            enable_mixed_chunk=False,
        )
        declare_resolution(
            server_args,
            "_handle_attention_backend_compatibility",
            disable_radix_cache=True,
        )


def handle_linear_attn_backend(server_args: Any):
    cfg = resolving_view(server_args)
    import torch

    # SM100+: default to FlashInfer GDN decode (and MTP verify, via pool API)
    # when the user hasn't explicitly chosen a decode backend and
    # mamba-ssm-dtype is bf16 (required by FlashInfer GDN on SM100+).
    # Fixed in FlashInfer v0.6.7: flashinfer-ai/flashinfer#2810
    if (
        cfg.linear_attn_decode_backend is None
        and cfg.linear_attn_backend != "helion"
        and get_platform().is_sm100
        and cfg.mamba_ssm_dtype == "bfloat16"
        # Stage 4: flashinfer's recurrent_kda compiles the state slot stride
        # as a free int64, so it reads the page-major/unified envelope-strided
        # state correctly — the unified-memory skip is no longer needed (the
        # page-major gate now allows flashinfer for linear-attn decode).
    ):
        declare_resolution(
            server_args,
            "_handle_linear_attn_backend",
            linear_attn_decode_backend="flashinfer",
        )
        logger.info(
            "SM100+ detected with mamba-ssm-dtype=bfloat16, "
            "defaulting --linear-attn-decode-backend to flashinfer."
        )

    # SM100+ FlashInfer GDN decode requires bf16 state; SM90 uses float32.
    decode = cfg.linear_attn_decode_backend or cfg.linear_attn_backend

    # FlashKDA is a prefill-only KDA kernel (no decode kernel) but shares the
    # backend choice list, so guard it from being selected for decode: error
    # on an explicit --linear-attn-decode-backend flashkda, and fall back to
    # triton decode when it was only inherited from base=flashkda (prefill
    # keeps FlashKDA).
    if decode == "flashkda":
        if cfg.linear_attn_decode_backend == "flashkda":
            raise ValueError(
                "--linear-attn-decode-backend flashkda is not supported: "
                "FlashKDA is prefill-only. Use "
                "--linear-attn-prefill-backend flashkda (decode stays on triton)."
            )
        declare_resolution(
            server_args,
            "_handle_linear_attn_backend",
            linear_attn_decode_backend="triton",
        )
        decode = "triton"
        logger.info(
            "FlashKDA is prefill-only; using triton for KDA decode "
            "(FlashKDA stays on prefill)."
        )

    if (
        decode == "flashinfer"
        and cfg.mamba_ssm_dtype != "bfloat16"
        and get_platform().is_cuda
        and torch.cuda.get_device_capability()[0] >= 10
    ):
        raise ValueError(
            "--linear-attn-decode-backend flashinfer on SM100+ requires "
            "--mamba-ssm-dtype bfloat16, "
            f"got {cfg.mamba_ssm_dtype!r}"
        )

    verify = cfg.linear_attn_verify_backend
    if verify is None and decode == "flashinfer":
        verify = "flashinfer"
    if (
        verify == "flashinfer"
        and cfg.mamba_ssm_dtype != "bfloat16"
        and get_platform().is_cuda
        and torch.cuda.get_device_capability()[0] >= 10
    ):
        raise ValueError(
            "--linear-attn-verify-backend flashinfer on SM100+ requires "
            "--mamba-ssm-dtype bfloat16, "
            f"got {cfg.mamba_ssm_dtype!r}"
        )

    # SM100+ FlashInfer GDN prefill requires CUDA 13+ (CuTe DSL kernel)
    # for correctness and best performance.
    prefill = cfg.linear_attn_prefill_backend or cfg.linear_attn_backend
    cuda_version = torch.version.cuda
    cuda_major = int(cuda_version.split(".")[0]) if cuda_version is not None else 0
    if (
        prefill == "flashinfer"
        and get_platform().is_cuda
        and torch.cuda.get_device_capability()[0] >= 10
        and cuda_major < 13
    ):
        raise ValueError(
            "--linear-attn-prefill-backend flashinfer on SM100+ requires CUDA 13+, "
            f"got CUDA {cuda_version or 'unknown'}"
        )

    # ReplaySSM buffered decode guards. Runs on Triton, or Helion for KDA.
    # cuda-graph is supported (slice 1b: CUDA-graph-safe static
    # write-cursor buffers). The RADIX prefix cache is now supported (slice
    # 2b: the decode kernel force-flushes the ring into temporal[slot] on
    # the radix track boundary `seq_lens % mamba_track_interval == 0`, and
    # the COW copy-into-slot path resets the ring cursor) -- so the
    # --disable-radix-cache requirement is dropped.
    #
    # Slice 2b only wires the no_buffer mamba scheduler strategy (the
    # default). The extra_buffer strategy donates the track snapshot via
    # `donate_mamba_ping_pong_slot` with a separate ping-pong slot swap that
    # does NOT route through MambaPool.copy_from, so the ReplaySSM ring
    # cursor of the donated/kept slot would not be reset there. Handling
    # that donation path is a follow-up; for now require no_buffer.
    if cfg.enable_linear_replayssm:
        if decode not in {"triton", "helion"}:
            raise ValueError(
                "--enable-linear-replayssm requires Triton, or Helion for "
                "KDA, as the linear-attn decode backend; got "
                f"--linear-attn-decode-backend={decode!r}."
            )

        if mamba_extra_buffer_of(resolved_view(server_args)):
            raise ValueError(
                "--enable-linear-replayssm requires --mamba-radix-cache-strategy "
                "no_buffer (the default); the extra_buffer ping-pong "
                "donation path is not yet supported (follow-up). Got "
                f"--mamba-radix-cache-strategy={cfg.mamba_radix_cache_strategy!r}."
            )
        if cfg.disaggregation_mode != "null":
            # The disaggregated decode pool (HybridMambaDecodeReqToTokenPool)
            # is not wired for the ReplaySSM ring, so the flag would silently
            # no-op there; disagg also runs a different cache/coordination
            # flow that is not yet validated for ReplaySSM (follow-up).
            raise ValueError(
                "--enable-linear-replayssm is not supported under PD "
                "disaggregation yet (follow-up). Got "
                f"--disaggregation-mode={cfg.disaggregation_mode!r}."
            )
        if cfg.linear_replayssm_cache_len < 1:
            raise ValueError(
                "--linear-replayssm-cache-len must be >= 1, got "
                f"{cfg.linear_replayssm_cache_len}."
            )

    # ReplaySSM spec-verify (Part B of #28511): linear-chain target verify via
    # fold-every-commit -- the verify stores each draft step's raw inputs into
    # the per-slot (rawv, rawk, g, beta) window and the commit replays the
    # accepted prefix into the fp32 checkpoint. The intra-window interaction
    # uses a strictly-lower causal mask, so it is valid ONLY for a linear
    # draft chain (speculative_eagle_topk in {None, 1}, i.e. NEXTN / MTP);
    # EAGLE tree verify (topk > 1) must fall back to the recurrent verify.
    # GDN sizes the window to the draft maximum; KDA (kda_backend) keeps a
    # --linear-replayssm-cache-len window and folds via its own fused
    # verify ring-write + commit_kda_replayssm_after_verify.
    if cfg.enable_linear_replayssm_spec:
        if cfg.speculative_eagle_topk not in (None, 1):
            raise ValueError(
                "--enable-linear-replayssm-spec requires a linear draft chain "
                "(--speculative-eagle-topk in {None, 1}); the chunked verify "
                "kernel uses a strictly-lower causal mask and is invalid for "
                "EAGLE tree verify. Got "
                f"--speculative-eagle-topk={cfg.speculative_eagle_topk!r}."
            )
        if decode not in ("triton", "flashinfer"):
            raise ValueError(
                "--enable-linear-replayssm-spec requires the triton or "
                "flashinfer linear-attn decode backend, got "
                f"--linear-attn-decode-backend={decode!r}."
            )
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        ragged_mode = read_ragged_verify_mode()
        if ragged_mode is not RaggedVerifyMode.STATIC:
            # Ragged ring-writes need the KDA fold-every-commit family
            # (DSPARK/DFLASH) + the triton verify kernel (nv_cutedsl falls
            # back to it for ragged layouts). The GDN ring-write kernels do
            # not take the ragged layout and the flashinfer verify kernel
            # never writes the ring -> a stale ring would be folded; keep
            # refusing those combinations.
            _algo = (cfg.speculative_algorithm or "").upper()
            verify = cfg.linear_attn_verify_backend
            if _algo not in ("DSPARK", "DFLASH") or verify not in (
                "triton",
                "nv_cutedsl",
            ):
                raise ValueError(
                    "--enable-linear-replayssm-spec with "
                    f"SGLANG_RAGGED_VERIFY_MODE={ragged_mode.value} requires the "
                    "KDA fold-every-commit family (DSPARK/DFLASH) and a "
                    "ring-writing verify kernel (--linear-attn-verify-backend "
                    "triton or nv_cutedsl); got "
                    f"algorithm={cfg.speculative_algorithm!r}, "
                    f"verify={verify!r}. Use SGLANG_RAGGED_VERIFY_MODE=static."
                )
        if cfg.disaggregation_mode == "prefill":
            raise ValueError(
                "--enable-linear-replayssm-spec is not supported on a PD "
                "prefill server: the ring is spec-verify-only scratch and "
                "the prefill server never runs spec verify."
            )
        if cfg.enable_linear_replayssm:
            raise ValueError(
                "--enable-linear-replayssm-spec and --enable-linear-replayssm are "
                "mutually exclusive: they share the ring storage but drive it "
                "with incompatible cursor protocols (per-decode-forward vs "
                "per-verify-commit advance)."
            )
        if cfg.mamba_ssm_dtype is None:
            logger.info(
                "--enable-linear-replayssm-spec: setting --mamba-ssm-dtype "
                "float32 (the closed-loop exact fold keeps the SSM checkpoint "
                "bit-identical to the recurrent baseline)."
            )
            declare_resolution(
                server_args,
                "_handle_linear_attn_backend",
                mamba_ssm_dtype="float32",
            )
        elif cfg.mamba_ssm_dtype != "float32":
            logger.warning(
                "--enable-linear-replayssm-spec with --mamba-ssm-dtype=%s: the "
                "closed-loop fold re-quantizes the committed state each "
                "commit/flush (fp32 keeps it bit-exact to the fp32 recurrent "
                "baseline), so it may drift over long sequences. Validate "
                "accuracy for your model.",
                cfg.mamba_ssm_dtype,
            )


def handle_multi_item_scoring(server_args: Any):
    """Setup and validate multi-item scoring constraints.

    Auto-disables settings incompatible with MIS mechanics (CUDA graph,
    radix cache, chunked prefill). Asserts on attention backend since
    changing it silently could surprise users who intentionally picked
    a non-flashinfer backend.
    """

    cfg = resolving_view(server_args)
    if not cfg.enable_mis:
        return

    if cfg.cuda_graph_config.decode.backend != Backend.DISABLED:
        logger.warning("CUDA graph is disabled because --enable-mis is set.")
    declare_resolution(
        server_args,
        "_handle_multi_item_scoring",
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
        ),
    )
    declare_resolution(
        server_args,
        "_handle_multi_item_scoring",
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )

    if not cfg.disable_radix_cache:
        logger.warning("Radix cache is disabled because --enable-mis is set.")
        declare_resolution(
            server_args,
            "_handle_multi_item_scoring",
            disable_radix_cache=True,
        )

    if cfg.chunked_prefill_size != -1:
        logger.warning("Chunked prefill is disabled because --enable-mis is set.")
        declare_resolution(
            server_args,
            "_handle_multi_item_scoring",
            chunked_prefill_size=-1,
        )

    prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
    assert prefill_backend == "flashinfer" and decode_backend == "flashinfer", (
        "Multi-item scoring requires flashinfer attention backend for custom attention mask support. "
        f"Please set --attention-backend flashinfer when using --enable-mis. "
        f"Current backends: prefill={prefill_backend}, decode={decode_backend}"
    )


def handle_deterministic_inference(server_args: Any):
    from sglang.srt.server_args import (
        RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND,
    )

    cfg = resolving_view(server_args)
    if cfg.rl_on_policy_target is not None:
        logger.warning("Enable deterministic inference because of rl_on_policy_target.")
        declare_resolution(
            server_args,
            "_handle_deterministic_inference",
            enable_deterministic_inference=True,
        )

        # For VLM
        envs.SGLANG_VLM_CACHE_SIZE_MB.set(0)
        # TODO remove this environment variable as a whole
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(True)

    if cfg.enable_deterministic_inference:
        if cfg.enable_aiter_allreduce_fusion:
            logger.warning(
                "Disable --enable-aiter-allreduce-fusion because deterministic inference is enabled."
            )
            declare_resolution(
                server_args,
                "_handle_deterministic_inference",
                enable_aiter_allreduce_fusion=False,
            )

        # Moved to the resolution pipeline (arg_groups/overrides.py:
        # _deterministic_allreduce_fusion_disable), invoked here at its
        # legacy slot.

        run_post_process_pass(server_args, _deterministic_allreduce_fusion_disable)

        # The forced-pytorch sampling write and the attention backend
        # fill/validation moved to the resolution pipeline
        # (arg_groups/overrides.py), invoked at their legacy slots.

        run_post_process_pass(server_args, _deterministic_sampling_backend)
        is_deepseek_model = False
        if parse_connector_type(cfg.model_path) != ConnectorType.INSTANCE:
            try:
                hf_config = model_config_of(server_args).hf_config
                model_arch = hf_config.architectures[0]
                is_deepseek_model = model_arch in [
                    "DeepseekV2ForCausalLM",
                    "DeepseekV3ForCausalLM",
                    "DeepseekV32ForCausalLM",
                    "MistralLarge3ForCausalLM",
                    "PixtralForConditionalGeneration",
                    "GlmMoeDsaForCausalLM",
                    "Glm4MoeLiteForCausalLM",
                    "Glm5NextForConditionalGeneration",
                ]
            except Exception:
                pass

        # Check attention backend
        run_post_process_pass(server_args, _deterministic_attention_backend)

        attention_backend = resolved_view(server_args).attention_backend
        if is_deepseek_model:
            if attention_backend not in RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND:
                raise ValueError(
                    f"Currently only {RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND} attention backends are supported for deterministic inference with absorbed-MLA models. But you're using {attention_backend}."
                )
            if attention_backend == "fa4" and not get_platform().is_sm100_or_sm110:
                raise ValueError(
                    "Deterministic inference with absorbed-MLA models on the fa4 "
                    "attention backend requires SM100/SM110: it runs "
                    "absorbed MLA, whose qv argument flash_attn.cute only "
                    "implements on those archs."
                )

        if attention_backend not in RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND:
            # Currently, only certain backends support radix cache. Support for other backends is in progress
            declare_resolution(
                server_args,
                "_handle_deterministic_inference",
                disable_radix_cache=True,
            )
            logger.warning(
                f"Currently radix cache is not compatible with {attention_backend} attention backend for deterministic inference. It will be supported in the future."
            )

        # Check TP size
        if cfg.tp_size > 1:
            if get_platform().is_hip:
                # AMD: use 1-stage all-reduce kernel which is inherently deterministic
                # (each GPU reads all data from all GPUs, reduces locally in fixed order)
                logger.info("AMD/ROCm: Using 1-stage all-reduce kernel (deterministic)")
            else:
                # CUDA: use NCCL tree algorithm
                os.environ["NCCL_ALGO"] = "allreduce:tree"
                # Not declared: set_default_server_args() writes this field
                # too, through its `args` parameter, so a declaration here
                # would be a second source for one field.
                declare_resolution(
                    server_args,
                    "_handle_deterministic_inference",
                    disable_custom_all_reduce=True,
                )
                # should_torch_symm_mem_allreduce() takes the
                # symmetric-memory path only below a byte threshold, so
                # which reduce runs would follow the token count.
                declare_resolution(
                    server_args,
                    "_handle_deterministic_inference",
                    enable_torch_symm_mem=False,
                )
                # Each channel carries a differently shaped tree and the
                # channel count is picked from the message size, so a
                # token's reduction order would follow the token count.
                nchannels = str(envs.SGLANG_DETERMINISTIC_NCCL_NCHANNELS.get())
                os.environ["NCCL_MIN_NCHANNELS"] = nchannels
                os.environ["NCCL_MAX_NCHANNELS"] = nchannels
                logger.warning(
                    "NCCL_ALGO is set to 'allreduce:tree', the NCCL channel count is pinned, and custom and symmetric-memory all reduce are disabled for deterministic inference when TP size > 1."
                )
