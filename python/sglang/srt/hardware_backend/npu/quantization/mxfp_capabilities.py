"""Local Ascend MXFP contracts, not a quantization or kernel dispatch registry.

Config/Method/Scheme still select the algorithm and own loading. KernelSpec is
reused for provider/device metadata; no SoC probing or provider selection lives
here. A successful preflight is not a hardware validation result.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable

from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
)

W8A8 = "W8A8_MXFP8"
W4A8 = "W4A8_MXFP"
W4A4 = "W4A4_MXFP4"
W4A4_DUAL = "W4A4_MXFP4_DUALSCALE"
MXFP_SCHEMES = (W8A8, W4A8, W4A4, W4A4_DUAL)


@dataclass(frozen=True)
class MXFPCapability:
    runtime: str
    layer_kind: str
    scheme: str
    source: str
    scale_mode: str
    checkpoint_dtype: str
    weight_dtype: str
    activation_dtype: str
    kernel: KernelSpec | None
    required_ops: tuple[str, ...]
    input_alignment: int = 1
    fallback: str | None = None
    blocker: str | None = None
    hardware_validation: str = "Not validated on this revision"

    @property
    def key(self) -> str:
        return f"{self.runtime}/{self.layer_kind}/{self.scheme}/{self.source}"

    @property
    def implementation(self) -> str:
        if self.blocker:
            return "blocked"
        if self.fallback:
            return "fallback"
        if (
            self.runtime == "diffusion"
            and self.scheme == W4A4
            and self.source == "online"
        ):
            return "implemented (experimental)"
        return "implemented"

    @property
    def scale_contract(self) -> str:
        if self.scale_mode == "dual":
            return "L0 FLOAT32 / block=512, L1 UE8M0 / block=32; not single-level UE8M0"
        return "UE8M0 bytes (+127), block=32; paired/transposed after load"

    @property
    def weight_layout(self) -> str:
        if self.fallback:
            return "Existing full-precision layout"
        if self.layer_kind == "moe":
            return "Expert weight/scale transpose; existing NZ/ND alignment fallback"
        if self.scale_mode == "dual":
            return "FRACTAL_NZ weight (not transposed); L0 scale transposed"
        if self.scheme == W4A8:
            return "FRACTAL_NZ packed weight transpose; scale [K/64,N,2]"
        if self.scheme == W4A4:
            return "Packed strided weight transpose; scale [K/64,N,2]"
        return "ND strided weight [K,N]; scale [ceil(K/64),N,2]"

    @property
    def fused_ops(self) -> tuple[str, ...]:
        # W4A8 retains its existing unfused path (e.g. SwiGLU clamp).
        return (
            ("npu_grouped_matmul_swiglu_quant_v2",)
            if self.layer_kind == "moe" and self.scheme == W4A8
            else ()
        )

    @property
    def parallelism(self) -> str:
        if self.layer_kind == "dense":
            return (
                "TP validates local K; EP not a Dense execution axis; sizes unverified"
            )
        return "Ascend TP and DeepEP normal/LL adapters; TP/EP sizes unverified"

    @property
    def hardware(self) -> str:
        if self.fallback:
            return "No MXFP operator needed for existing full-precision fallback"
        return (
            "Ascend 950 for dual-level MXFP4"
            if self.scale_mode == "dual"
            else "Ascend NPU with the listed MXFP ops; generations unverified"
        )


def _entry(
    runtime, kind, scheme, source, method, *, dual=False, fallback=None, blocker=None
):
    w8 = scheme == W8A8
    weight_dtype = "float8_e4m3fn" if w8 else "float4_e2m1fn_x2"
    activation_dtype = "float8_e4m3fn" if w8 or scheme == W4A8 else "float4_e2m1fn_x2"
    if fallback:
        weight_dtype = activation_dtype = "BF16/FP16 (existing fallback)"
    checkpoint_dtype = (
        "BF16/FP16"
        if source == "online"
        else "float8_e4m3fn [N,K]"
        if w8
        else "packed uint8 [N,K/2]"
    )
    if kind == "moe" and source == "modelslim":
        checkpoint_dtype = checkpoint_dtype.replace("[N,", "[E,N,")
    if kind == "dense":
        ops = (
            ("npu_dynamic_dual_level_mx_quant", "npu_dual_level_quant_matmul")
            if dual
            else ("npu_dynamic_mx_quant", "npu_quant_matmul")
        )
    else:
        ops = ("npu_dynamic_mx_quant", "npu_grouped_matmul")
        if w8:
            ops += ("npu_grouped_matmul_swiglu_quant_v2",)
    # Diffusion MXFP8 preserves partial blocks and pads odd scale counts before
    # delegating to the shared kernel. Other ModelSlim loaders still need K=64
    # alignment. These are loader restrictions, not hardware support claims.
    partial_scales = (
        runtime == "diffusion" and kind == "dense" and w8 and source == "modelslim"
    )
    alignment = (
        1
        if fallback or partial_scales
        else 64
        if source == "modelslim" and not dual
        else 2
        if not w8
        else 1
    )
    kernel = (
        None
        if fallback
        else KernelSpec(
            op=f"quantization.ascend_{scheme.lower()}_{kind}",
            backend=KernelBackend.TORCH_NPU,
            target=method
            if method.startswith("sglang.")
            else f"sglang.srt.hardware_backend.npu.quantization.{method}",
            capabilities=frozenset({CapabilityRequirement.NPU}),
            format_signature=FormatSignature(
                supported_dtypes=(weight_dtype, activation_dtype),
                description="L0/L1 dual-level" if dual else "MX block=32, UE8M0 scale",
            ),
        )
    )
    return MXFPCapability(
        runtime,
        kind,
        scheme,
        source,
        "dual" if dual else "single",
        checkpoint_dtype,
        weight_dtype,
        activation_dtype,
        kernel,
        () if fallback else ops,
        alignment,
        fallback,
        blocker,
    )


# Inventory only paths present in this checkout. Online W4 MoE methods on other
# feature branches must not be advertised as implemented on this baseline.
MXFP_CAPABILITIES = (
    _entry(
        "srt", "dense", W8A8, "online", "linear_method_npu:NPUMXFP8LinearMethod.apply"
    ),
    _entry(
        "srt",
        "dense",
        W8A8,
        "modelslim",
        "linear_method_npu:NPUMXFP8LinearMethod.apply",
    ),
    _entry(
        "srt",
        "dense",
        W4A8,
        "online",
        "linear_method_npu:NPUMXFP4W4A8LinearMethod.apply",
    ),
    _entry(
        "srt",
        "dense",
        W4A8,
        "modelslim",
        "linear_method_npu:NPUMXFP4W4A8OfflineLinearMethod.apply",
    ),
    _entry(
        "srt",
        "dense",
        W4A4,
        "online",
        "linear_method_npu:NPUDualLevelMXFP4LinearMethod.apply",
        dual=True,
    ),
    _entry(
        "srt",
        "dense",
        W4A4,
        "modelslim",
        "linear_method_npu:NPUSingleLevelMXFP4OfflineLinearMethod.apply",
    ),
    _entry("srt", "moe", W8A8, "online", "moe_methods:NPUMXFP8MoEMethod.apply"),
    _entry("srt", "moe", W8A8, "modelslim", "moe_methods:NPUMXFP8MoEMethod.apply"),
    _entry("srt", "moe", W4A8, "modelslim", "moe_methods:NPUW4A8MXFP4MoEMethod.apply"),
    _entry("srt", "moe", W4A4, "modelslim", "moe_methods:NPUW4A4MXFP4MoEMethod.apply"),
    _entry(
        "srt",
        "moe",
        W4A8,
        "online",
        "",
        fallback="No quantized online W4A8 MoE method on this baseline; retain full-precision BF16/FP16 MoE",
    ),
    _entry(
        "srt",
        "moe",
        W4A4,
        "online",
        "",
        fallback="No quantized online W4A4 MoE method on this baseline; retain full-precision BF16/FP16 MoE",
    ),
    _entry(
        "diffusion",
        "dense",
        W8A8,
        "online",
        "linear_method_npu:NPUMXFP8LinearMethod.apply",
    ),
    _entry(
        "diffusion",
        "dense",
        W8A8,
        "modelslim",
        "sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp8_scheme:ModelSlimMXFP8Scheme.apply_weights",
    ),
    _entry(
        "diffusion",
        "dense",
        W4A4,
        "online",
        "sglang.multimodal_gen.runtime.layers.quantization.mxfp4_npu:NPUMXFP4DiffusionLinearMethod.apply",
        dual=True,
    ),
    _entry(
        "diffusion",
        "dense",
        W4A4,
        "modelslim",
        "linear_method_npu:NPUSingleLevelMXFP4OfflineLinearMethod.apply",
    ),
    _entry(
        "diffusion",
        "dense",
        W4A4_DUAL,
        "modelslim",
        "sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp4_scheme:ModelSlimMXFP4Scheme.apply_weights",
        dual=True,
        blocker="Diffusion ModelSlim dual-scale constructor references undefined is_dual_scale_key on this baseline; fix that path before enabling it",
    ),
)


def get_mxfp_capability(
    runtime: str, layer_kind: str, scheme: str, source: str
) -> MXFPCapability:
    for cap in MXFP_CAPABILITIES:
        if (cap.runtime, cap.layer_kind, cap.scheme, cap.source) == (
            runtime,
            layer_kind,
            scheme,
            source,
        ):
            return cap
    raise ValueError(
        f"Ascend MXFP: no implementation for {runtime}/{layer_kind}/{scheme}/{source}; no fallback will be added"
    )


def validate_mxfp_configuration(
    cap: MXFPCapability,
    *,
    prefix="",
    platform: PlatformInfo | None = None,
    input_size: int | None = None,
    runner: str | None = None,
    a2a_backend: str | None = None,
    swiglu_limit: float | None = None,
    activation: str | None = None,
) -> str | None:
    """Check only known metadata; return an existing fallback reason, never select one."""
    where = f"Ascend MXFP {cap.key} ({prefix or 'layer'})"
    if cap.blocker:
        raise ValueError(f"{where}: {cap.blocker}")
    if cap.fallback:
        return cap.fallback
    if platform is not None and not cap.kernel.is_available(platform):
        raise ValueError(
            f"{where}: requires NPU / {cap.kernel.backend.value}, got {platform.device_type}"
        )
    if input_size is not None and (input_size <= 0 or input_size % cap.input_alignment):
        raise ValueError(
            f"{where}: local input K={input_size} must be positive and divisible by {cap.input_alignment}; current {cap.source} loader/operator contract, no BF16 fallback"
        )
    if cap.layer_kind == "moe":
        if runner is not None and runner not in ("auto", "ascend"):
            raise ValueError(
                f"{where}: requires moe-runner-backend auto/ascend, got {runner}"
            )
        if a2a_backend == "ascend_fuseep":
            raise ValueError(
                f"{where}: ascend_fuseep uses a different INT8/float-scale weight contract, not MXFP"
            )
        if (
            cap.scheme in (W8A8, W4A8)
            and activation is not None
            and activation != "silu"
        ):
            raise ValueError(
                f"{where}: fused MXFP gmm1 requires SiLU, got {activation}"
            )
        if cap.scheme == W8A8 and swiglu_limit is not None and swiglu_limit > 0:
            raise ValueError(
                f"{where}: MXFP8 fused gmm1 has no clamp or unfused path; swiglu_limit={swiglu_limit} is unsupported"
            )
    return None


def validate_mxfp_loaded_weights(cap: MXFPCapability, weight, scale) -> None:
    """Validate raw ModelSlim tensors before reshape/cast or any NPU operation."""
    if cap.source != "modelslim":
        raise ValueError(
            "Loaded checkpoint validation is only for ModelSlim, not online quantization"
        )
    if weight is None or scale is None:
        raise ValueError(
            f"Ascend MXFP {cap.key}: checkpoint requires weight and weight_scale"
        )
    expected_dtype = "float8_e4m3fn" if cap.scheme == W8A8 else "uint8"
    if (
        str(weight.dtype).removeprefix("torch.") != expected_dtype
        or str(scale.dtype).removeprefix("torch.") != "uint8"
    ):
        raise ValueError(
            f"Ascend MXFP {cap.key}: checkpoint requires {expected_dtype} weight and uint8 UE8M0 scale, got {weight.dtype}/{scale.dtype}; do not requantize/repack"
        )
    ndim = 3 if cap.layer_kind == "moe" else 2
    if (
        weight.ndim != ndim
        or scale.ndim != ndim
        or tuple(weight.shape[:-1]) != tuple(scale.shape[:-1])
    ):
        raise ValueError(
            f"Ascend MXFP {cap.key}: incompatible checkpoint weight/scale shapes {tuple(weight.shape)}/{tuple(scale.shape)}"
        )
    k = weight.shape[-1] * (1 if cap.scheme == W8A8 else 2)
    validate_mxfp_configuration(cap, input_size=k)
    if scale.shape[-1] != (k + 31) // 32:
        raise ValueError(
            f"Ascend MXFP {cap.key}: expected ceil(K/32)={((k + 31) // 32)} UE8M0 blocks, got {scale.shape[-1]}; partial blocks must not be dropped"
        )


def require_mxfp_operators(
    cap: MXFPCapability,
    available_ops: Iterable[str] | None = None,
    *,
    fused: bool = False,
) -> None:
    """Late preflight: ops must be registered by torch_npu before post-load starts."""
    if cap.fallback:
        return
    required_ops = cap.required_ops + (cap.fused_ops if fused else ())
    if available_ops is None:
        import torch

        from sglang.srt.utils import is_npu

        validate_mxfp_configuration(cap, platform=PlatformInfo.detect())

        if (
            cap.weight_dtype == "float4_e2m1fn_x2"
            and cap.scale_mode == "single"
            and is_npu()
        ):
            import torch_npu

            if not isinstance(getattr(torch_npu, "float4_e2m1fn_x2", None), int):
                raise ValueError(
                    f"Ascend MXFP {cap.key}: requires torch_npu.float4_e2m1fn_x2 packed FP4 enum; a torch dtype is not the NPU API enum"
                )
        available_ops = {op for op in required_ops if hasattr(torch.ops.npu, op)}
        if (
            cap.scale_mode == "single"
            and getattr(torch, "float8_e8m0fnu", None) is None
        ):
            raise ValueError(
                f"Ascend MXFP {cap.key}: torch.float8_e8m0fnu is required for UE8M0 scales"
            )
    missing = set(required_ops) - set(available_ops)
    if missing:
        raise ValueError(
            f"Ascend MXFP {cap.key}: required NPU operators not registered: {', '.join(sorted(missing))}; check torch_npu/CANN, not a generation fallback"
        )


def render_mxfp_support_table() -> str:
    """The same rows consumed by preflight also drive docs and the CPU matrix."""
    rows = [
        "<table>",
        "<thead><tr><th>Path</th><th>Implementation</th><th>Checkpoint / compute</th><th>Scale / local K</th><th>Requirements / limits</th><th>Hardware evidence</th></tr></thead>",
        "<tbody>",
    ]
    for cap in MXFP_CAPABILITIES:
        fields = (
            cap.key,
            cap.implementation,
            f"{cap.checkpoint_dtype}; W={cap.weight_dtype}, A={cap.activation_dtype}; {cap.weight_layout}",
            f"{cap.scale_mode}: {cap.scale_contract}; K % {cap.input_alignment} = 0",
            f"{cap.hardware}; ops={', '.join(cap.required_ops) or 'none'}; fused-only ops={', '.join(cap.fused_ops) or 'none'}; {cap.parallelism}; {cap.blocker or cap.fallback or 'existing routing constraints still apply'}",
            cap.hardware_validation,
        )
        rows.append(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in fields) + "</tr>"
        )
    rows.extend(("</tbody>", "</table>"))
    return "\n".join(rows)
