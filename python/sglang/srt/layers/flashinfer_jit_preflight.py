"""Preflight and poisoned-cache recovery for flashinfer's JIT sampling ops.

Flashinfer JIT-compiles its sampling kernels (``top_k_renorm_probs`` /
``top_p_renorm_probs`` / ...) on first use per process. On environments where
that compile fails (e.g. the bundled-cccl vs pip-toolkit header mismatch in
sglang#43412), the failure currently surfaces mid-request from deep inside the
speculative verify path, escapes the scheduler event loop, and takes the whole
server down. The failed build also leaves partial artifacts on disk
(``build.ninja`` + object files, no ``sampling.so``), so every restart walks
into the same failure again.

This module provides the containment primitives:

- :func:`flashinfer_sampling_build_dir` — locate the on-disk JIT build dir for
  flashinfer's (single) sampling module.
- :func:`is_flashinfer_sampling_poisoned` — detect the partial-build state
  left behind by a failed compile (artifacts present, ``sampling.so`` absent,
  no concurrent build holding the lock).
- :func:`clear_poisoned_flashinfer_sampling_cache` — targeted recovery:
  delete only the sampling op's build dir so the next warm-up retries clean.
- :func:`preflight_flashinfer_sampling_ops` — warm the sampling ops with tiny
  dummy tensors at startup so any compile failure surfaces before serving,
  instead of on the first non-greedy request.

Notes on flashinfer's layout (verified against flashinfer 0.6.18,
``flashinfer/jit/env.py`` and ``flashinfer/jit/core.py``):

- All sampling ops share ONE JitSpec named ``"sampling"``, so one warm-up call
  covers renorm + sampling + softmax for the whole process.
- Cache root: ``$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer`` (default
  ``~/.cache/flashinfer``), then ``<version>/<cuda-archs>/cached_ops/``.
- The JIT ``.so`` is ``cached_ops/sampling/sampling.so`` — a sibling of
  ``build.ninja``; objects are ``csrc_*.cuda.o``.
- A failed ninja build raises ``RuntimeError("Ninja build failed. ...")`` with
  the compiler log in the message; flashinfer itself has no per-op cache
  clear, no poison detection, and no retry logic.
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)

_SAMPLING_SPEC_NAME = "sampling"
_SAMPLING_LIBRARY_NAME = f"{_SAMPLING_SPEC_NAME}.so"


def _get_flashinfer_jit_env():
    """Import flashinfer's JIT env module defensively.

    Returns None when flashinfer (or its JIT machinery) is unavailable, e.g.
    on CPU-only builds — callers treat that as "nothing to preflight".
    """
    try:
        from flashinfer.jit import env as jit_env

        return jit_env
    except Exception:
        return None


def flashinfer_sampling_build_dir() -> str | None:
    """Return the on-disk JIT build dir for flashinfer's sampling module.

    The layout is ``<cache>/<version>/<archs>/cached_ops/sampling``; returns
    None when flashinfer's JIT env cannot be imported.
    """
    jit_env = _get_flashinfer_jit_env()
    if jit_env is None:
        return None
    try:
        jit_dir = jit_env.FLASHINFER_JIT_DIR
    except AttributeError:
        return None
    if jit_dir is None:
        return None
    return os.path.join(str(jit_dir), _SAMPLING_SPEC_NAME)


def _sampling_lock_available(build_dir: str) -> bool:
    """Best-effort check that no other process is building right now.

    flashinfer serializes builds with a filelock at
    ``<jit_dir>/tmp/sampling.lock``. A build in progress by another process
    leaves the same on-disk state as a poisoned cache (artifacts, no .so), so
    we only declare poison when the lock can be taken immediately.
    """
    lock_path = os.path.join(
        os.path.dirname(os.path.normpath(build_dir)),
        "tmp",
        f"{_SAMPLING_SPEC_NAME}.lock",
    )
    try:
        from filelock import FileLock

        lock = FileLock(lock_path)
        lock.acquire(timeout=0)
        lock.release()
        return True
    except Exception:
        # filelock missing, lock held by a concurrent build, or the lock dir
        # does not exist yet — treat as "do not touch the dir".
        return False


def is_flashinfer_sampling_poisoned(build_dir: str | None = None) -> bool:
    """Detect the partial-build state left by a failed JIT compile.

    True iff the sampling build dir exists, has ninja artifacts (``build.ninja``
    or any object file), has no linked ``sampling.so``, and no concurrent
    build holds the JIT lock. This mirrors what a failed
    ``RuntimeError: Ninja build failed`` leaves behind (sglang#43412).
    """
    if build_dir is None:
        build_dir = flashinfer_sampling_build_dir()
    if build_dir is None or not os.path.isdir(build_dir):
        return False

    has_library = os.path.isfile(os.path.join(build_dir, _SAMPLING_LIBRARY_NAME))
    if has_library:
        return False

    has_ninja_file = os.path.isfile(os.path.join(build_dir, "build.ninja"))
    has_objects = any(name.endswith(".o") for name in os.listdir(build_dir))
    if not (has_ninja_file or has_objects):
        return False

    return _sampling_lock_available(build_dir)


def clear_poisoned_flashinfer_sampling_cache(
    build_dir: str | None = None,
) -> tuple[bool, str | None]:
    """Delete a poisoned sampling-op build dir so the next warm-up retries clean.

    This is deliberately targeted (only ``cached_ops/sampling/``) — flashinfer's
    own ``clear_cache_dir()`` wipes every cached op for the current
    version+arch, which would throw away unrelated working kernels.

    Returns (cleared, path): cleared is True when a poisoned dir was removed.
    Never touches a dir that looks clean, already has ``sampling.so``, or is
    being built by another process right now.
    """
    if build_dir is None:
        build_dir = flashinfer_sampling_build_dir()
    if build_dir is None:
        return False, None
    if not is_flashinfer_sampling_poisoned(build_dir):
        return False, build_dir

    logger.warning(
        "flashinfer sampling JIT cache is in a partial-build state (failed "
        "compile left artifacts without %s); clearing %s so warm-up can "
        "retry from scratch",
        _SAMPLING_LIBRARY_NAME,
        build_dir,
    )
    shutil.rmtree(build_dir, ignore_errors=True)
    return True, build_dir


def preflight_flashinfer_sampling_ops(device: str = "cuda") -> tuple[bool, str]:
    """Warm flashinfer's sampling ops with tiny dummy tensors.

    Compiling (or loading) the sampling module here means any JIT failure
    surfaces at startup with the full compiler log, instead of on the first
    non-greedy request. Because all sampling ops share one JitSpec, one call
    warms renorm + sampling + softmax for the process.

    Returns (ok, detail): ok False + detail carries the failure message
    (caller decides whether to raise or fall back). The dummy renorm call
    also proves the freshly built module actually loads and launches, which a
    bare compile would not.
    """
    clear_poisoned_flashinfer_sampling_cache()

    try:
        import torch
        from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs
    except Exception as e:
        return False, f"flashinfer sampling import failed: {e}"

    try:
        # Tiny tensors: the point is to trigger (or load) the JIT build and
        # launch the op once, not to benchmark. float32 + a small vocab keeps
        # the call valid on every supported arch.
        probs = torch.full((1, 8), 1.0 / 8, dtype=torch.float32, device=device)
        top_ks = torch.full((1, 1), 4, dtype=torch.int32, device=device)
        top_ps = torch.full((1, 1), 0.9, dtype=torch.float32, device=device)
        top_k_renorm_probs(probs, top_ks)
        top_p_renorm_probs(probs, top_ps)
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"flashinfer sampling op warm-up failed: {e}"

    return True, "flashinfer sampling ops warmed successfully"
