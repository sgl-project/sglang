"""Attention quantization scheme for offline MXFP4 checkpoints on Ascend NPU.

:class:`AttentionQuantScheme` is the per-attention-block quantization plan
derived from the checkpoint's ``quant_model_description.json``: which
precision each denoising step runs at, whether the exported Q/K rotation
matrices must be registered and applied, and which attention backends
implement the quantized kernel. The model layer, the attention backend, and
the Cache-DiT precision guard all read from this object instead of
re-deriving it from raw descriptor dicts.

Shared, non-attention-specific helpers (timestep precision resolution, FP4
packing, MindIE quantization defaults) live in ``mxfp4_utils.py``.
"""

import weakref

import torch

from sglang.multimodal_gen.runtime.layers.quantization.mxfp4_utils import (
    resolve_precision,
)
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

    def register_qk_rotation_buffers(
        self,
        module: torch.nn.Module,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> bool:
        """Register the exported ``q_rot``/``k_rot`` buffers on ``module``.

        Centralizes what every model with exported rotations needs: register
        the two ``[head_dim, head_dim]`` buffers (so they load from the
        checkpoint and appear in its state dict) and :meth:`bind` the module
        for :meth:`apply_qk_rotation`. No-op returning ``False`` when the
        checkpoint does not export both rotations; ``True`` when they were
        registered. Models whose attention cannot apply the rotation (e.g.
        sparse variants) should reject the configuration on ``True``.
        """
        if not self.use_offline_qk_rotation:
            return False
        for name in ("q_rot", "k_rot"):
            module.register_buffer(
                name,
                torch.empty(head_dim, head_dim, dtype=dtype),
                persistent=True,
            )
        self.bind(module)
        return True

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
