# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Compatibility fingerprint of a graph artifact (design section 11).

A saved artifact is usable only by a process whose fingerprint is *equal* to
the saved one. Fingerprints are compared field by field and by the digest of
their deterministic JSON encoding, never by Python hash, and the first
mismatching field is logged (the weight-cache daemon precedent). The device
UUID is deliberately excluded so artifacts move between identical GPUs and DP
replicas.

Nothing in this module touches the GPU; ``compute_fingerprint`` is the one
entry point that will, and it is a stub in this draft.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import msgspec

_DETERMINISTIC_JSON = msgspec.json.Encoder(order="deterministic")


class EnvironmentFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    """Software and device environment. Any change here invalidates the
    artifact by design (design section 16, version drift)."""

    torch: str = ""
    cuda_runtime: str = ""
    driver: str = ""
    sgl_kernel: str = ""
    flashinfer: str = ""
    triton: str = ""
    deepgemm: str = ""
    device_name: str = ""
    compute_capability: str = ""
    sm_count: int = 0
    total_memory: int = 0
    allocator_config: str = ""
    cuda_graph_env: dict[str, str] = {}


class ModelFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    hf_config_sha256: str = ""
    context_length: int = 0
    dtype: str = ""
    kv_cache_dtype: str = ""
    quantization: Optional[str] = None
    page_size: int = 0
    spec: dict[str, str] = {}


class ParallelFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    """Every ``ParallelState`` size and rank; artifacts are strictly per rank
    (design section 9.4)."""

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    ep_size: int = 1
    attn_cp_size: int = 1
    dcp_size: int = 1
    moe_dp_size: int = 1
    nnodes: int = 1
    node_rank: int = 0
    pdmux: bool = False


class GeometryFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    """Per-runner shape space, taken *after* capture-time config mutation
    (``PhaseConfig.bs`` and ``full_prefill_max_req`` are resolved in place)."""

    runner: str
    capture_sizes: tuple[int, ...] = ()
    captured_req_width: int = 1
    forward_mode: str = ""
    hidden_mode: str = ""
    seq_len_fill_value: int = 0
    req_to_token_shape: tuple[int, int] = (0, 0)
    max_total_num_tokens_before_resize: int = 0
    max_total_num_tokens_after_resize: int = 0
    shared_logits_rows: int = 0
    vocab_size: int = 0


class BackendFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    graph_backend_decode: str = ""
    graph_backend_prefill: str = ""
    attention_decode: str = ""
    attention_prefill: str = ""
    moe_runner: str = ""
    moe_a2a: str = ""
    shared_experts_fusion: bool = False
    all_reduce_impl: str = ""
    all_reduce_thresholds: dict[str, int] = {}
    enable_symm_mem: bool = False
    fusion_backend: str = ""
    dp_padding_mode: str = ""
    tbo: bool = False
    lora: bool = False
    dsa: bool = False


class WeightCacheFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    """Weight provenance (design section 10). ``cache_config`` carries the
    daemon ``CacheConfig`` verbatim as a dict because the live object is not
    hashable; a silent disk fallback in client mode changes ``mode`` and
    fails the comparison."""

    mode: str = "off"
    transport: str = ""
    cache_config: dict[str, Any] = {}
    preloaded_weights_bytes: int = 0
    daemon_generation: str = ""


class GraphArtifactFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    """The complete compatibility record stored in every ``RankManifest``."""

    environment: EnvironmentFingerprint = EnvironmentFingerprint()
    model: ModelFingerprint = ModelFingerprint()
    parallel: ParallelFingerprint = ParallelFingerprint()
    geometry: tuple[GeometryFingerprint, ...] = ()
    backends: BackendFingerprint = BackendFingerprint()
    weight_cache: WeightCacheFingerprint = WeightCacheFingerprint()
    weight_layout_digest: str = ""
    kernel_image_set_digest: str = ""
    autotune_cache_digest: str = ""
    placement: str = "relocate"
    format_version: int = 1


def fingerprint_digest(fingerprint: GraphArtifactFingerprint) -> str:
    """sha256 of the deterministic JSON encoding; names the artifact directory."""
    return hashlib.sha256(_DETERMINISTIC_JSON.encode(fingerprint)).hexdigest()


def _walk_diff(saved: Any, live: Any, path: str, out: list[str]) -> None:
    if isinstance(saved, dict) and isinstance(live, dict):
        for key in sorted(set(saved) | set(live)):
            child = f"{path}.{key}" if path else str(key)
            if key not in saved or key not in live:
                out.append(child)
                continue
            _walk_diff(saved[key], live[key], child, out)
        return
    if isinstance(saved, list) and isinstance(live, list):
        if len(saved) != len(live):
            out.append(f"{path}[len]")
            return
        for index, (a, b) in enumerate(zip(saved, live)):
            _walk_diff(a, b, f"{path}[{index}]", out)
        return
    if saved != live:
        out.append(path)


def diff_fingerprints(
    saved: GraphArtifactFingerprint, live: GraphArtifactFingerprint
) -> list[str]:
    """Dotted paths of every leaf that differs, in sorted order.

    An empty list means the two fingerprints are equal. Callers log the first
    entry so an operator sees *why* an artifact was rejected instead of a bare
    digest mismatch.
    """
    out: list[str] = []
    _walk_diff(msgspec.to_builtins(saved), msgspec.to_builtins(live), "", out)
    return out


def compute_fingerprint(
    *,
    placement: str,
    geometry: tuple[GeometryFingerprint, ...] = (),
) -> GraphArtifactFingerprint:
    """Collect the live fingerprint after capture (design section 11).

    Sources, in the order the design lists them: the weight-cache
    ``CacheConfig`` and transport; every ``ParallelState`` size and rank;
    the model config digest; the resolved post-capture ``CudaGraphConfig``
    and per-runner geometry; the attention, MoE and collective backend
    selections; torch, CUDA, driver and kernel-library versions plus the
    device name and compute capability; the weight layout digest over sorted
    ``(name, region, offset, nbytes, dtype, shape, stride)``; the kernel image
    set digest; and the flashinfer autotune cache digest.
    """
    raise NotImplementedError(
        "compute_fingerprint: collecting the live environment, model, parallel "
        "and backend fields is not implemented in this draft; see "
        "DESIGN_cuda_graph_serialization.md section 11"
    )
