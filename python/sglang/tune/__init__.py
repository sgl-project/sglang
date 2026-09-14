"""Attune — automated per-device attention-backend tuning for SGLang.

v1 scope (this prototype): choose the ``(prefill_backend, decode_backend, page_size)``
triple per (device, model attention profile) empirically, and auto-ingest it at
engine init behind ``--enable-attune`` (off by default). It *refines* SGLang's
existing ``_get_default_attn_backend`` heuristic — it never bypasses the capability
gate, only overrides the pick among gate-surviving candidates.

The design mirrors SGLang's own in-tree precedents: the MoE tuned-config pipeline
(``tuning_fused_moe_triton.py`` + ``get_moe_configs``, originated in PR #2628) and
``flashinfer_autotune.py`` (device-keyed, fingerprinted, capability-gated).

Everything here runs in ``--mock`` mode with no GPU and no SGLang installed, so the
architecture is exercisable end-to-end; the real-hardware paths are guarded and slot
in when this package sits inside a live SGLang tree.
"""

__version__ = "0.1.0-prototype"

from .loader import get_attune_config, pick_backends  # noqa: F401
from .shapes import AttnProfile, DecodeShape, PrefillShape  # noqa: F401
