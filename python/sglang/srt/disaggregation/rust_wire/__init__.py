"""Rust wire-codec backend for the P/D disaggregation data plane.

Selected by SGLANG_DISAGG_WIRE_BACKEND ("python" default | "rust"). The
Rust codecs are byte-identical to the Python builders — the golden vectors
in rust/sglang-disagg/tests/vectors/wire.json are generated FROM the Python
code — so mixed rust/python fleets interoperate and rolling upgrades are
safe; the backend only changes who burns CPU on encode/decode.

`rust` is the loaded extension module, or None when the python backend is
selected or the extension is unavailable (logged fallback).
"""

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _load():
    backend = envs.SGLANG_DISAGG_WIRE_BACKEND.get()
    if backend == "python":
        return None
    if backend != "rust":
        logger.warning(
            "unknown SGLANG_DISAGG_WIRE_BACKEND=%r, expected 'python' or 'rust'; "
            "using python",
            backend,
        )
        return None
    try:
        from sglang.srt.disaggregation.rust_wire.extension import bindings
    except Exception as e:
        logger.warning("Rust disagg wire codecs unavailable (%s); using python", e)
        return None
    logger.info("Using Rust disagg wire codecs")
    return bindings


rust = _load()
