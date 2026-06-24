# SPDX-License-Identifier: Apache-2.0
"""Native FP8 DiT dispatch for OmniDreams.

Routes the per-chunk OmniDreams DiT forward through FlashDreams' native
``optimized_dit_forward`` (FP8 tensor-core GEMMs + FP8 flash/SageAttention-3/
SpargeAttn attention + FP8 AdaLN-LoRA/modulate), reusing the vendored native tree
under ``multimodal_gen/native/omnidreams_singleview/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
# Native loading (GPU/sm_120 only)                                            #
# --------------------------------------------------------------------------- #
@dataclass
class _NativeHandles:
    extension: Any
    optimized_dit: Any
    cosmos_fp8_utils: Any


def _load_native(mode: str = "auto") -> _NativeHandles | None:
    if mode == "disabled":
        return None

    try:
        from sglang.multimodal_gen.native import load_extension
        from sglang.multimodal_gen.native.singleview_loader import load_python_module

        ext = load_extension()
        if ext is None:
            if mode == "required":
                from sglang.multimodal_gen.native import NativeAccelerationUnavailable
                raise NativeAccelerationUnavailable(
                    "native FP8 DiT extension is unavailable (sm_120 build required)"
                )
            return None
        if not hasattr(ext, "optimized_dit_forward"):
            raise RuntimeError("native ext missing optimized_dit_forward")
        return _NativeHandles(
            extension=ext,
            optimized_dit=load_python_module("optimized_dit"),
            cosmos_fp8_utils=load_python_module("cosmos_fp8_utils"),
        )
    except Exception as e:
        if mode == "required":
            raise
        logger.info("OmniDreams FP8 DiT native ext unavailable (%s); using eager DiT.", e)
        return None


# --------------------------------------------------------------------------- #
# FP8 weight preparation (CPU-runnable)                                       #
# --------------------------------------------------------------------------- #
def _unfuse_self_attn_qkv_for_cosmos(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Split fused ``self_attn.to_qkv`` back into q/k/v for the vendored Cosmos prep.

    The sglang OmniDreamsDiT stores self-attention Q/K/V as a single
    ``MergedColumnParallelLinear`` (``to_qkv`` = ``cat([q, k, v], dim=0)``,
    q/k/v row order -- see ``OmniDreamsAttention.__init__``), but the vendored
    Cosmos weight-prep / FP8-quant helpers
    (``prepare_cosmos_streaming_weights``, ``quantize_cosmos_fp8_weights``)
    expect the legacy split ``q_proj``/``k_proj``/``v_proj`` keys. This splits
    the fused weight back into three equal row shards so those helpers run
    unchanged and produce byte-identical artifacts to the pre-merge path:
    per-output-channel FP8 scales are row-independent, so quantizing the fused
    tensor then splitting == splitting then quantizing (verified EXACT on the
    30 self-attn QKV tensors).

    The fused ``to_qkv`` key is dropped from the output: the native FP8 runtime
    consumes the rebuilt ``qkv_proj`` and does not read ``to_qkv``, so keeping
    it would only ship ~1.3 GB of dead bf16 in the artifact.
    """
    suffix = "self_attn.to_qkv.weight"
    fused_keys = [k for k in state_dict if k.endswith(suffix)]
    if not fused_keys:
        return state_dict
    out = dict(state_dict)
    for fused_key in fused_keys:
        prefix = fused_key[: -len(suffix)]  # "blocks.{i}."
        fused = out.pop(fused_key)
        if fused.dim() != 2 or fused.shape[0] % 3 != 0:
            raise ValueError(
                f"cannot split {fused_key!r} (shape {tuple(fused.shape)}) into "
                "3 equal q/k/v row shards"
            )
        q_weight, k_weight, v_weight = torch.chunk(fused, 3, dim=0)
        out[f"{prefix}self_attn.q_proj.weight"] = q_weight.contiguous()
        out[f"{prefix}self_attn.k_proj.weight"] = k_weight.contiguous()
        out[f"{prefix}self_attn.v_proj.weight"] = v_weight.contiguous()
    return out


def prepare_fp8_dit_weights(
    state_dict: dict[str, torch.Tensor],
    num_blocks: int,
    *,
    cosmos_fp8_utils: Any | None = None,
    linear_policy: str = "all",
) -> dict[str, torch.Tensor]:
    if cosmos_fp8_utils is None:
        from sglang.multimodal_gen.native.singleview_loader import load_python_module
        cosmos_fp8_utils = load_python_module("cosmos_fp8_utils")
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    # sglang's DiT fuses self-attn Q/K/V into ``to_qkv``; the vendored Cosmos
    # prep/quant helpers expect the legacy split q/k/v keys. Unfuse first so
    # they run unchanged (see _unfuse_self_attn_qkv_for_cosmos).
    cpu_state = _unfuse_self_attn_qkv_for_cosmos(cpu_state)
    return cosmos_fp8_utils.prepare_cosmos_quantized_streaming_weights(
        cpu_state, num_blocks=num_blocks, device=None, linear_policy=linear_policy,
    )


# --------------------------------------------------------------------------- #
# FP8 → bf16 dequantization (weight-only FP8, Ideogram 4 style)               #
# --------------------------------------------------------------------------- #
_WEIGHT_SUFFIX = ".weight"
_SCALE_SUFFIX = ".weight_scale"


def dequantize_fp8_weights_to_bf16(
    fp8_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Dequantize FP8 quantized weights back to bf16 for standard PyTorch inference.

    The FP8 artifact stores each quantized linear weight as:
      - ``K.weight``: raw E4M3 bytes (torch.uint8, [out, in])
      - ``K.weight_scale``: per-output-channel scale (float16, [out])

    This function reverses the quantization:
      weight_bf16 = weight.view(float8_e4m3fn).to(float32) * scale.unsqueeze(1)

    Returns a dict mapping original ``.weight`` keys to dequantized bf16 tensors,
    suitable for ``model.load_state_dict(strict=False)``.
    """
    result: dict[str, torch.Tensor] = {}
    dequant_count = 0

    for key, value in fp8_weights.items():
        # Already dequantized or not a weight — pass through
        if key.endswith(_SCALE_SUFFIX):
            continue  # scales consumed by their weight key
        if key.endswith(".weight") and value.dtype == torch.uint8:
            # FP8 raw E4M3 bytes — dequantize
            scale_key = key + "_scale"
            if scale_key in fp8_weights:
                scale = fp8_weights[scale_key]
                weight_f32 = value.view(torch.float8_e4m3fn).to(torch.float32)
                dequant = (weight_f32 * scale.to(torch.float32).unsqueeze(1)).to(
                    torch.bfloat16
                )
                result[key] = dequant.contiguous()
                dequant_count += 1
            else:
                logger.warning("FP8 weight %s has no scale %s; skipping", key, scale_key)
        elif key.endswith(".weight_prepared"):
            # bf16 transposed prepared weight — skip (model uses .weight layout)
            continue
        elif key.endswith("_fp8_prepared"):
            # FP8 prepared alias — skip (base .weight key covers it)
            continue
        elif key.endswith("_fp8_prepared_scale"):
            # FP8 prepared alias scale — skip
            continue
        else:
            # Non-FP8 key (biases, norms, embeddings, bf16 weights) — pass through
            if key not in result:
                result[key] = value

    logger.info(
        "Dequantized %d FP8 weight pairs to bf16 (%d total keys in result)",
        dequant_count,
        len(result),
    )
    return result


# --------------------------------------------------------------------------- #
# Adapter: present SGLang's OmniDreamsDiT to the vendored executor            #
# --------------------------------------------------------------------------- #
class _NetCfg:
    def __init__(self, arch: Any) -> None:
        self.num_blocks = int(arch.num_blocks)
        self.num_heads = int(arch.num_heads)
        self.model_channels = int(arch.model_channels)
        self.adaln_lora_dim = int(getattr(arch, "adaln_lora_dim", 256))
        self.timestep_scale = float(getattr(arch, "timestep_scale", 1.0))
        self.patch_temporal = int(getattr(arch, "patch_temporal", 1))
        self.patch_spatial = int(getattr(arch, "patch_spatial", 2))
        self.use_crossattn_projection = True


class _ExecConfig:
    def __init__(self, arch: Any, len_t: int, dtype: torch.dtype) -> None:
        self.network = _NetCfg(arch)
        self.num_views = 1
        self.len_t = int(len_t)
        self.dtype = dtype
        self.use_cuda_graph: bool = False
        self.cuda_graph_warmup_iters: int = 0


class _SGLTransformerAdapter:
    """Presents SGLang's OmniDreamsDiT as the FlashDreams transformer.

    SGLang handles patch/unpatch and cache-init at the stage level, so the
    vendored ``_release_network_after_fp8_snapshot`` (which replaces
    ``self.network`` with a lightweight shape-ops stub) is short-circuited
    here via ``_released_network_for_fp8``.  This keeps the adapter pointing
    at the real ``OmniDreamsDiT`` for the lifetime of the process.
    """

    def __init__(self, sgl_dit: Any, arch: Any, len_t: int, dtype: torch.dtype,
                 height: int, width: int) -> None:
        self.network = sgl_dit
        self.config = _ExecConfig(arch, len_t, dtype)
        self._output_height = int(height)
        self._output_width = int(width)
        self._use_cuda_graph = False
        self._cuda_graph_capture_ar_idx = 0

    @property
    def _released_network_for_fp8(self) -> bool:
        return True

    @_released_network_for_fp8.setter
    def _released_network_for_fp8(self, value: bool) -> None:
        pass

    def _maybe_inject_image(self, latent, cache):
        return latent

    def _select_mask(self, cache):
        return None


# --------------------------------------------------------------------------- #
# Public FP8 DiT dispatcher                                                    #
# --------------------------------------------------------------------------- #
class OmniDreamsFP8DiT:
    """Per-chunk FP8 DiT forward via the native ``optimized_dit_forward``."""

    def __init__(self, sgl_dit: Any, arch: Any, native: _NativeHandles, *,
                 attention_backend: str = "auto",
                 dit_backend: str = "fp8_kvcache_cudnn",
                 fp8_prepared_weights: dict | None = None) -> None:
        self._native = native
        self._arch = arch
        self._sgl_dit = sgl_dit
        self._attention_backend = attention_backend
        self._dit_backend = dit_backend
        self._executor: Any | None = None
        # Pre-quantized weights (CPU dict from torch.load). Held as a CPU ref
        # during init (DiT may still be offloaded). Moved to GPU inside
        # _ensure_executor at first __call__, when the device is known.
        self._prepared_weights = fp8_prepared_weights

    @staticmethod
    def _move_prepared_weights_to_device(
        prepared: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Move the CPU FP8 weight dict to GPU, preserving storage sharing.

        ``cosmos_fp8_utils.add_cosmos_fp8_prepared_aliases`` sets each
        ``{key}_fp8_prepared`` alias to ``weight.contiguous()`` -- i.e. the
        *same bytes* as the canonical ``{key}`` (same for the scale alias).
        A naive ``{k: v.to(device).contiguous() for k, v in ...}`` moves each
        tensor independently and breaks that storage sharing, duplicating the
        ~1.6 GiB of FP8 weights (and the C++ bridge reads the alias, not the
        canonical). Preserve sharing by moving the canonical first and pointing
        the alias at the same on-GPU tensor.
        """
        alias_suffixes = ("_fp8_prepared", "_fp8_prepared_scale")
        moved: dict[str, Any] = {}
        for k, v in prepared.items():
            if not isinstance(v, torch.Tensor):
                moved[k] = v
                continue
            if k.endswith(alias_suffixes):
                # Strip the longer suffix first so "_fp8_prepared_scale" is
                # handled before the "_fp8_prepared" substring it ends with.
                canonical_key = k.removesuffix("_fp8_prepared_scale").removesuffix(
                    "_fp8_prepared"
                )
                base = moved.get(canonical_key)
                if (
                    isinstance(base, torch.Tensor)
                    and base.dtype == v.dtype
                    and base.shape == v.shape
                    and base.is_contiguous()
                ):
                    moved[k] = base  # share storage with the moved canonical
                    continue
            moved[k] = v.to(device=device).contiguous()
        return moved

    def _ensure_executor(self, len_t: int, height: int, width: int) -> Any:
        if self._executor is None:
            adapter = _SGLTransformerAdapter(
                self._sgl_dit, self._arch, len_t, torch.bfloat16, height, width
            )
            self._executor = self._native.optimized_dit.OptimizedDiTExecutor(
                adapter,
                self._native.extension,
                dit_backend=self._dit_backend,
                attention_backend=self._attention_backend,
            )
            # Inject pre-quantized weights so _ensure_weights_snapshot()
            # finds them on first call and skips all quantization work.
            assert self._prepared_weights is not None, \
                "FP8 prepared weights missing; build_fp8_dit must inject them"
            device = next(self._sgl_dit.parameters()).device
            self._executor._optimized_weights = self._move_prepared_weights_to_device(
                self._prepared_weights, device
            )
            self._prepared_weights = None  # release CPU ref

            # Release the live BF16 DiT params from GPU VRAM. After the executor
            # is built and the FP8 weights + cross-attn KV are materialized, the
            # native C++ forward consumes only ``_optimized_weights`` + cache
            # tensors; the AR loop's only other use of the DiT is
            # ``patchify``/``unpatchify`` (pure einops reshapes, no parameter
            # reads). This mirrors flashdreams' ``_release_network_after_fp8_snapshot``
            # intent: the ~4.1 GiB of BF16 params are dead weight once the FP8
            # snapshot exists. We move them to CPU (not drop the object) so the
            # stage's ``self.transformer`` reference stays valid for
            # patchify/unpatchify. No ``empty_cache()``: the caching allocator
            # reuses the freed block for the C++ forward's workspace/KV, which
            # is what actually lowers peak VRAM.
            if envs.SGLANG_OMNIDREAMS_FP8_RELEASE_LIVE_DIT:
                self._sgl_dit.cpu()
        return self._executor

    def release_executor(self) -> None:
        """Drop the native FP8 executor + its on-GPU weight snapshot.

        After the AR rollout the executor (``_optimized_weights`` ~2 GiB of
        FP8 weights + the native workspace/KV template refs) is dead -- the
        DecodingStage runs the Wan VAE, which does not touch the DiT. Releasing
        it before the decode peak lowers fp8 peak VRAM toward eager. Mirrors
        the flashdreams "release after fp8 snapshot" intent at the stage
        boundary. Idempotent.
        """
        if self._executor is not None:
            # Drop the injected FP8 weight dict first so it frees even if the
            # C++ side retains a ref to the executor object.
            self._executor._optimized_weights = None
        self._executor = None
        self._sgl_dit = None
        self._native = None

    @torch.inference_mode()
    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        condition_video_input_mask: torch.Tensor,
        rope_freqs: torch.Tensor,
        hdmap_condition: torch.Tensor | None,
        kv_caches: list,
        cross_attn_kv: list,
        view_indices: torch.Tensor | None,
        ar_idx: int,
        len_t: int,
        hp: int,
        wp: int,
    ) -> torch.Tensor:
        opt = self._native.optimized_dit
        ex = self._ensure_executor(len_t, hp * self._arch.patch_spatial,
                                   wp * self._arch.patch_spatial)
        compute_write_start = opt.compute_self_attn_write_start

        B, L, D = hidden_states.shape
        T, HW = len_t, hp * wp
        ext_dtype = ex.config.dtype
        noisy = hidden_states.reshape(B, 1, T, HW, D).to(ext_dtype)
        mask = condition_video_input_mask.reshape(
            B, 1, T, HW, condition_video_input_mask.shape[-1]
        ).to(ext_dtype)
        hdmap_in = (
            None if hdmap_condition is None
            else hdmap_condition.reshape(B, 1, T, HW, hdmap_condition.shape[-1]).to(ext_dtype)
        )

        timestep_b = timestep.reshape(1).expand(B).contiguous()
        inv = ex._ensure_invariant_tensors(ar_idx=ar_idx, timesteps=timestep_b)
        rope_cos, rope_sin = ex._ensure_rope_tensors(ar_idx=ar_idx, rope_freqs=rope_freqs)
        ex._ensure_weights_snapshot()

        k_self = [c._k for c in kv_caches]
        v_self = [c._v for c in kv_caches]
        n_heads = self._arch.num_heads
        head_dim = self._arch.model_channels // n_heads
        k_cross = [kv[0].reshape(kv[0].shape[0], -1, n_heads, head_dim)
                   for kv in cross_attn_kv]
        v_cross = [kv[1].reshape(kv[1].shape[0], -1, n_heads, head_dim)
                   for kv in cross_attn_kv]
        write_start = compute_write_start(kv_caches[0])

        hdmap_embed = ex._ensure_hdmap_tensor(ar_idx=ar_idx, input_for_ext=hdmap_in)
        hdmap_for_ext = (
            hdmap_in if hdmap_embed is None
            else ex._empty_hdmap_tensor(device=noisy.device, dtype=ext_dtype)
        )

        # Inject runtime config + workspace (mirrors _ensure_fp8_runtime which
        # is normally bypassed in the SGLang fast path — see call trace doc).
        ex._resolve_runtime_attention_backend(noisy.device)
        use_sage3 = ex._attention_backend == "sage3_fp8"
        use_sparge = ex._attention_backend == "sparge"
        attn_backend = ex._attention_backend if (use_sage3 or use_sparge) else "fp8_cudnn"
        tokens = int(T * HW)
        # max_attn_tokens = max sequence length across all KV caches (self + cross)
        max_attn_tokens = tokens
        for tlist in (k_self, k_cross):
            for t in tlist:
                max_attn_tokens = max(max_attn_tokens, int(t.size(1)))
        workspace = self._native.optimized_dit._make_cosmos_streaming_workspace(
            batch=int(k_self[0].size(0)),
            tokens=tokens,
            max_attn_tokens=max_attn_tokens,
            num_blocks=int(self._arch.num_blocks),
            model_channels=int(self._arch.model_channels),
            heads=n_heads,
            ff=int(4 * self._arch.model_channels),
            lora_dim=int(getattr(self._arch, "adaln_lora_dim", 256)),
            device=noisy.device,
            dtype=torch.bfloat16,
            use_sage3_fp8_attention=use_sage3,
        )
        # Build KV cache tensors in the layout the C++ bridge expects.
        # Mirrors vendored ``_ensure_fp8_runtime`` (optimized_dit.py:1203-1269).
        extra_config: dict[str, Any] = {}
        k_cross_fp8: list[torch.Tensor] = []
        v_cross_fp8: list[torch.Tensor] = []

        if use_sage3:
            # Sage3 FP8 attention needs FP4 cross-attn KV + scale factors.
            sage3_cross = [
                self._native.extension.sage3_quantize_cross_kv_bf16(k, v)
                for k, v in zip(k_cross, v_cross, strict=True)
            ]
            extra_config["k_cross_sage3_fp4_caches"] = [
                tensors[0] for tensors in sage3_cross
            ]
            extra_config["v_cross_sage3_fp4_caches"] = [
                tensors[1] for tensors in sage3_cross
            ]
            extra_config["k_cross_sage3_sf_caches"] = [
                tensors[2] for tensors in sage3_cross
            ]
            extra_config["v_cross_sage3_sf_caches"] = [
                tensors[3] for tensors in sage3_cross
            ]
            # k/v_cross_fp8 stay empty: Sage3 consumes FP4 directly.
        else:
            # Generic FP8 attention path: cuDNN FP8 or Sparge consumes FP8 KV.
            k_cross_fp8 = [
                t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                for t in k_cross
            ]
            v_cross_fp8 = [
                t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                for t in v_cross
            ]

        extra_config["k_cross_fp8_caches"] = k_cross_fp8
        extra_config["v_cross_fp8_caches"] = v_cross_fp8
        # FP8 self-KV cache: the C++ bridge writes the current chunk's K/V at
        # [write_start, write_start+M) and self-attention reads [0, write_start+M).
        # The AR-context prefix [0, write_start) must already hold the previous
        # chunks' K/V. The bf16 self-cache (k_self/v_self) carries the correctly
        # rolled AR context (maintained by the pipeline via
        # BlockKVCache.before_update), so re-quantize it into the fp8 cache each
        # call. (flashdreams persists + left-rolls the fp8 cache directly; this
        # stateless fast path re-quantizes from bf16 instead -- same result with
        # no cross-sequence state and no accumulating fp8 roll error.)
        extra_config["k_self_fp8_caches"] = [
            t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous() for t in k_self
        ]
        extra_config["v_self_fp8_caches"] = [
            t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous() for t in v_self
        ]
        extra_config["_last_rolled"] = {}
        trace_tensor = None
        if os.environ.get("SGLANG_OMNIDREAMS_FP8_TRACE"):
            trace_tensor = torch.empty(
                (int(self._arch.num_blocks), 4, B, tokens, D),
                device=noisy.device,
                dtype=torch.bfloat16,
            )
            extra_config["cosmos_trace_tensor"] = trace_tensor

        ex._apply_runtime_config({
            "cosmos_linear_backend": "fp8",
            "cosmos_attention_backend": attn_backend,
            "cosmos_kv_cache_backend": "fp8",
            "cosmos_quantized_prepared": True,
            "cosmos_quantized_prepared_strict": True,
            "cosmos_workspace": workspace,
            "cosmos_attn_tc_scale_is_ones": True,
            **extra_config,
        })

        out = ex._predict_flow_ext_impl(
            noisy, mask, hdmap_for_ext, hdmap_embed, timestep_b, rope_freqs,
            inv.t_emb, inv.t_emb_silu, inv.adaln_lora, inv.final_shift, inv.final_scale,
            rope_cos, rope_sin, inv.block_mods_sa, inv.block_mods_ca, inv.block_mods_mlp,
            k_cross, v_cross, k_self, v_self, write_start,
        )
        if trace_tensor is not None:
            self._log_trace_std(trace_tensor)
        return out.reshape(B, L, out.shape[-1]).to(hidden_states.dtype)

    @staticmethod
    def _log_trace_std(trace_tensor: torch.Tensor) -> None:
        """Log per-block std-dev of the native FP8 trace (debug-only)."""
        trace_stats = trace_tensor.float().std(dim=(2, 3, 4)).detach().cpu()
        labels = ("sa", "ca", "mlp", "block")
        rows = []
        for block_idx in range(trace_stats.shape[0]):
            stats = " ".join(
                f"{labels[j]}={float(trace_stats[block_idx, j]):.4f}"
                for j in range(trace_stats.shape[1])
            )
            rows.append(f"b{block_idx:02d} {stats}")
        logger.info("OmniDreams FP8 native trace std: %s", " | ".join(rows))


def build_fp8_dit(
    sgl_dit: Any,
    arch: Any,
    *,
    mode: str,  # Required: "auto" | "disabled" | "required"
    fp8_prepared_path: str | None = None,
    attention_backend: str = "auto",
) -> OmniDreamsFP8DiT | None:
    """Build native FP8 DiT wrapper.

    Args:
        mode: "auto" (try, fallback on failure), "required" (raise on failure),
              "disabled" (return None).
        fp8_prepared_path: Path to pre-quantized FP8 weights (.pt from the
            offline exporter). Required for ``mode="required"``; for
            ``mode="auto"`` the caller falls back to eager DiT when the file
            is missing.
    """
    if mode == "disabled":
        return None
    native = _load_native(mode)
    if native is None:
        return None

    fp8_weights = None
    if fp8_prepared_path and os.path.exists(fp8_prepared_path):
        payload = torch.load(fp8_prepared_path, map_location="cpu", weights_only=True)
        fp8_weights = payload["weights"]  # meta validation done by caller (DenoisingStage)

    if fp8_weights is None:
        if mode == "required":
            raise FileNotFoundError(
                f"FP8 prepared weights not found at {fp8_prepared_path!r}. "
                "Run: python -m sglang.multimodal_gen.tools."
                "export_omnidreams_fp8_dit_weights "
                "--checkpoint <path> --output <path>"
            )
        # auto mode without file → return None, DenoisingStage falls back to eager
        return None

    return OmniDreamsFP8DiT(
        sgl_dit,
        arch,
        native,
        attention_backend=attention_backend,
        fp8_prepared_weights=fp8_weights,
    )
