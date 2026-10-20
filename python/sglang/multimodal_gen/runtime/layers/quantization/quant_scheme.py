"""Shared quantization scheme for offline MXFP checkpoints on Ascend NPU.

This module consolidates everything that more than one consumer needs from an
offline checkpoint's ``quant_model_description.json``:

- :class:`AttentionQuantScheme` -- the per-attention-block quantization plan:
  which precision each denoising step runs at, whether the exported Q/K
  rotation matrices must be registered and applied, and which attention
  backends implement the quantized kernel. The model layer, the attention
  backend, and the Cache-DiT precision guard all read from this object instead
  of re-deriving it from raw descriptor dicts.
- Timestep precision helpers (:func:`resolve_precision`,
  :func:`default_mxfp4_policy`) shared by attention and linear schemes.
- :func:`pack_fp4_weight` used by the weight loaders and ``wan_repack.py``.

Naming note: the checkpoints are produced by the ``msmodelslim`` tool and
detected via ``--quantization modelslim``; the scheme classes here are
deliberately named after the quantization format (MXFP4), not the tool.
"""

import weakref

import torch
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# msmodelslim names the exported rotations after the to_q/to_k projections
# (`to_q_rot`/`to_k_rot`); earlier exports used `q_rot`/`k_rot`. The Wan weight
# mapping (configs/models/dits/wanvideo.py) accepts both spellings, so
# descriptor lookups must too -- a missed lookup leaves the rotation buffers
# unregistered and the tensors are silently skipped at load.
_ROTATION_DESCRIPTOR_NAMES = (("to_q_rot", "q_rot"), ("to_k_rot", "k_rot"))

# Descriptors that previously selected FP8 attention. FP8 attention was
# removed: it only ever worked against MindIE's own kernels, which SGLang does
# not ship. Checkpoints exporting it must be re-quantized to MXFP4.
_UNSUPPORTED_ATTENTION_DESCRIPTORS = {
    "FP8_DYNAMIC": (
        "FP8 attention is no longer supported (it requires custom MindIE "
        "kernels); re-quantize this checkpoint with MXFP4 attention or keep "
        "attention in BF16"
    ),
}


def offline_qk_rotation_flags(description: dict, prefix: str) -> tuple[bool, bool]:
    """(q_exported, k_exported): whether ``prefix``'s Q/K rotations are FLOAT."""
    q_names, k_names = _ROTATION_DESCRIPTOR_NAMES
    return (
        any(description.get(f"{prefix}.{name}") == "FLOAT" for name in q_names),
        any(description.get(f"{prefix}.{name}") == "FLOAT" for name in k_names),
    )


def default_mxfp4_policy(num_steps: int) -> dict[str, str]:
    """MindIE-SD's default per-step W4A4/W4A8 schedule for single-level MXFP4.

    The first two and last two denoising steps run W4A8 (higher activation
    precision at the schedule boundaries); everything between runs W4A4. A
    40-step run uses W4A8 at zero-based steps 0, 1, 38, and 39.
    """
    return {
        str(step): "W4A8"
        for step in range(num_steps)
        if step < 2 or step >= num_steps - 2
    }


def resolve_precision(
    description: dict, target: str, default: str, default_policy: dict | None = None
) -> str:
    """Resolve the effective precision for ``target`` at the current timestep.

    ``target`` is a policy family in the checkpoint's ``timestep_policy``
    section (``fa``, ``w4a4_linear``, ``w8a8_linear``). Per-step entries win
    over a ``default`` entry, which wins over ``default_policy`` (the caller's
    schedule), which wins over ``default``.
    """
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context_or_none,
    )

    context = get_forward_context_or_none()
    policy = description.get("timestep_policy", {}).get(target, {})
    if not policy and not default_policy:
        return default
    if context is None:
        raise RuntimeError(f"{target} timestep policy requires a forward context")
    step = context.current_timestep
    fallback = (default_policy or {}).get(str(step), default)
    precision = policy.get(str(step), policy.get("default", fallback))
    allowed = {
        "fa": ("FLOAT", "MXFP4"),
        "w4a4_linear": ("W4A4", "W4A8"),
        "w8a8_linear": ("W8A8", "FLOAT"),
    }
    if precision not in allowed[target]:
        raise ValueError(f"Unsupported {target} precision: {precision}")
    return precision


def mxfp4_quant_kwargs(description: dict) -> dict:
    # Match MindIE-SD f1b15b4 C7 quantization defaults.
    return {
        "scale_alg": description.get("mxfp4_scale_alg", 2),
        "dst_type_max": description.get("mxfp4_dst_type_max", 7.25),
    }


def pack_fp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack one linear weight into two-FP4-per-byte uint8.

    Legacy msmodelslim exports store the FP4 values as a numeric
    ``float8_e4m3fn`` container ``[out, in]``; newer exports and the repack
    tool emit packed ``uint8`` ``[out, in/2]``. The loader expects packed.
    """
    if weight.dtype == torch.uint8:
        return weight
    if weight.dtype != torch.float8_e4m3fn or weight.shape[-1] % 2:
        raise ValueError(
            "MXFP4 weights must be packed uint8 or even-width FP8 containers"
        )
    # Convert numeric E2M1 values before the loader casts to the target dtype.
    values = weight.float()
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    codes = torch.searchsorted(magnitudes, values.abs()).clamp(max=7)
    if not torch.equal(magnitudes[codes], values.abs()):
        raise ValueError(
            "FP8 container contains values that are not representable as FP4 E2M1"
        )
    codes = (codes | (torch.signbit(values).long() << 3)).to(torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


class AttentionQuantScheme:
    """Attention quantization plan derived from the checkpoint descriptor.

    One instance per attention block. Owns the decisions that used to be
    scattered across the model layer, the attention backend, and the Cache-DiT
    guard:

    - which precision the block runs at for the current denoising step
      (:meth:`resolve_precision`),
    - whether the checkpoint's exported Q/K rotation matrices must be
      registered and applied (:attr:`use_offline_qk_rotation`,
      :meth:`apply_qk_rotation`),
    - which attention backends implement the quantized kernel
      (:attr:`quant_attention_backends`).

    The class is backend-agnostic by design: supporting quantized attention on
    a new backend means implementing the kernel there and adding the backend to
    :attr:`quant_attention_backends` -- no model-layer changes.
    """

    # Backends with a quantized (MXFP4) attention kernel. FA implements the
    # Ascend 950 kernel today; extend this set when another backend adds one.
    quant_attention_backends: frozenset[AttentionBackendEnum] = frozenset(
        {AttentionBackendEnum.FA}
    )

    def __init__(
        self,
        prefix: str,
        scheme: str | None,
        description: dict,
        q_rot_exported: bool,
        k_rot_exported: bool,
    ) -> None:
        self.prefix = prefix
        self.scheme = scheme
        self.description = description
        self.q_rot_exported = q_rot_exported
        self.k_rot_exported = k_rot_exported
        self._module_ref = None

    @classmethod
    def from_quant_config(cls, quant_config, prefix: str) -> "AttentionQuantScheme":
        """Build the scheme for ``prefix`` (either the block or ``*.impl``).

        Attention descriptors may live under the plain prefix, the
        ``.quant_type`` suffix, or the ``.fa_q.scale`` suffix depending on the
        export; the first hit wins. A missing or ``FLOAT`` descriptor means the
        block runs unquantized attention.
        """
        prefix = prefix.removesuffix(".impl")
        description = getattr(quant_config, "quant_description", {}) or {}
        descriptor = next(
            (
                description[key]
                for key in (prefix, prefix + ".quant_type", prefix + ".fa_q.scale")
                if key in description
            ),
            None,
        )
        if descriptor in _UNSUPPORTED_ATTENTION_DESCRIPTORS:
            raise ValueError(
                f"Unsupported attention descriptor for {prefix}: {descriptor}. "
                + _UNSUPPORTED_ATTENTION_DESCRIPTORS[descriptor]
            )
        if descriptor is not None and descriptor != "FLOAT":
            if descriptor != "MXFP4_DYNAMIC":
                raise ValueError(
                    f"Unsupported attention descriptor for {prefix}: {descriptor}"
                )
            scheme = "MXFP4"
        else:
            scheme = None
        q_rot_exported, k_rot_exported = offline_qk_rotation_flags(description, prefix)
        return cls(prefix, scheme, description, q_rot_exported, k_rot_exported)

    @property
    def is_quant(self) -> bool:
        return self.scheme is not None

    @property
    def use_offline_qk_rotation(self) -> bool:
        return self.q_rot_exported and self.k_rot_exported

    @property
    def requires_exported_rotations(self) -> bool:
        """Whether skipping the checkpoint's q_rot/k_rot tensors is an error.

        Quantized attention is calibrated against the exported rotations, so a
        checkpoint that ships them while the model runs quantized attention
        must register them. Unquantized attention may skip them (any
        orthogonal matrix preserves scores).
        """
        return self.is_quant

    @property
    def required_backend(self) -> AttentionBackendEnum | None:
        """The backend to require on the attention layer, if any."""
        if not self.is_quant:
            return None
        if not self.quant_attention_backends:
            raise RuntimeError(
                f"No attention backend implements {self.scheme} quantized "
                f"attention for {self.prefix}"
            )
        return next(iter(self.quant_attention_backends))

    # -- Timestep precision policy ------------------------------------------

    @property
    def precision_policy_target(self) -> str | None:
        """Policy family for :func:`resolve_precision`; ``None`` when the
        block runs unquantized attention and has no per-step policy."""
        return "fa" if self.is_quant else None

    @property
    def precision_policy_default(self) -> str:
        return self.scheme or "FLOAT"

    def resolve_precision(self, default_policy: dict | None = None) -> str:
        return resolve_precision(
            self.description,
            self.precision_policy_target,
            self.precision_policy_default,
            default_policy,
        )

    # -- Q/K rotation handling ----------------------------------------------

    def bind(self, module) -> None:
        """Bind the module holding the registered ``q_rot``/``k_rot`` buffers.

        The buffers stay on the module (they must load from the checkpoint and
        appear in its state dict); the scheme only keeps a weak reference so it
        can move them to the activation device/dtype lazily.
        """
        self._module_ref = weakref.ref(module)

    def should_apply_rotation(
        self,
        backend: AttentionBackendEnum | None,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        legacy_mxfp8_enabled: bool = False,
    ) -> bool:
        """Whether this call needs the exported Q/K rotations applied.

        Quantized steps always rotate (calibration expects it). The legacy
        online-MXFP8 path (no descriptor scheme) rotates only on the FA
        backend's quantized shapes, mirroring its kernel constraints.
        """
        if not self.use_offline_qk_rotation:
            return False
        if self.is_quant:
            return self.resolve_precision() != "FLOAT"
        return (
            legacy_mxfp8_enabled
            and backend is AttentionBackendEnum.FA
            and query.shape[1:3] == key.shape[1:3]
            and key.shape == value.shape
            and (query.shape[0] * query.shape[1]) % 64 == 0
        )

    def apply_qk_rotation(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        module = self._module_ref() if self._module_ref is not None else None
        if module is None:
            raise RuntimeError(
                f"{self.prefix}: rotation buffers requested before bind()"
            )
        # Persist the device/dtype move on the buffers to avoid repeated
        # host-to-device copies on later steps.
        module.q_rot = module.q_rot.to(device=query.device, dtype=query.dtype)
        module.k_rot = module.k_rot.to(device=key.device, dtype=key.dtype)
        return torch.matmul(query, module.q_rot), torch.matmul(key, module.k_rot)
