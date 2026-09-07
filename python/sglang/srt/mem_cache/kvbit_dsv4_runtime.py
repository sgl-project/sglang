"""Fail-fast capability checks; no extension import on native-KV paths."""

from __future__ import annotations

from importlib import import_module
from typing import NamedTuple


class DSV4KVBitRuntimeCapability(NamedTuple):
    direct_packed_write: bool
    direct_packed_decode: bool


def require_dsv4_kvbit_runtime_capability(
    capability: DSV4KVBitRuntimeCapability | None = None,
) -> None:
    if capability is None:
        try:
            import_module("triton")
            wrapper = import_module("sgl_kernel.kvbit_flash_mla")
            wrapper.require_kvbit_int4_extension()
        except Exception as exc:
            raise RuntimeError(
                "--kv-cache-dtype int4 requires Triton and the built-in "
                "DSV4 INT4 AOT extension with a registered CUDA decode op. "
                "Rebuild kvbit_flashmla_ops for this checkout; "
                "native/scratch fallback is disabled."
            ) from exc
        return

    missing = []
    if not capability.direct_packed_write:
        missing.append("direct packed write")
    if not capability.direct_packed_decode:
        missing.append("direct packed decode")
    if missing:
        raise RuntimeError(
            "--kv-cache-dtype int4 requires DSV4 "
            + " and ".join(missing)
            + " capability; native/scratch fallback is disabled."
        )
