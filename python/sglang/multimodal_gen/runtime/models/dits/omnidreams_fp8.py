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
    return cosmos_fp8_utils.prepare_cosmos_quantized_streaming_weights(
        cpu_state, num_blocks=num_blocks, device=None, linear_policy=linear_policy,
    )


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
            self._executor._optimized_weights = {
                k: v.to(device=device).contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in self._prepared_weights.items()
            }
            self._prepared_weights = None  # release CPU ref
        return self._executor

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
        # Map Python-side attention names to C++ bridge names
        _PY_TO_CPP_ATTN = {"cudnn": "cudnn_bf16"}
        attn_backend = _PY_TO_CPP_ATTN.get(ex._attention_backend, ex._attention_backend)
        # KV cache backend: FP8 for sage3_fp8 (native FP8 attention),
        # BF16 for cudnn_bf16 (cuDNN FMHA with BF16 KV cache).
        use_fp8_kv = attn_backend == "sage3_fp8"
        kv_backend = "fp8" if use_fp8_kv else "bf16"
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
            use_sage3_fp8_attention=(attn_backend == "sage3_fp8"),
        )
        # Build KV cache tensors in the layout the C++ bridge expects.
        # Mirrors vendored ``_ensure_fp8_runtime`` (optimized_dit.py:1203-1269).
        use_sage3 = attn_backend == "sage3_fp8"
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
        elif use_fp8_kv:
            # Sparge / generic FP8 attention path.
            k_cross_fp8 = [
                t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                for t in k_cross
            ]
            v_cross_fp8 = [
                t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                for t in v_cross
            ]

        if use_fp8_kv:
            extra_config["k_cross_fp8_caches"] = k_cross_fp8
            extra_config["v_cross_fp8_caches"] = v_cross_fp8
            extra_config["k_self_fp8_caches"] = [
                torch.zeros_like(t, dtype=torch.uint8) for t in k_self
            ]
            extra_config["v_self_fp8_caches"] = [
                torch.zeros_like(t, dtype=torch.uint8) for t in v_self
            ]

        ex._apply_runtime_config({
            "cosmos_linear_backend": "fp8",
            "cosmos_attention_backend": attn_backend,
            "cosmos_kv_cache_backend": kv_backend,
            "cosmos_quantized_prepared": True,
            "cosmos_workspace": workspace,
            **extra_config,
        })

        out = ex._predict_flow_ext_impl(
            noisy, mask, hdmap_for_ext, hdmap_embed, timestep_b, rope_freqs,
            inv.t_emb, inv.t_emb_silu, inv.adaln_lora, inv.final_shift, inv.final_scale,
            rope_cos, rope_sin, inv.block_mods_sa, inv.block_mods_ca, inv.block_mods_mlp,
            k_cross, v_cross, k_self, v_self, write_start,
        )
        return out.reshape(B, L, out.shape[-1]).to(hidden_states.dtype)


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
