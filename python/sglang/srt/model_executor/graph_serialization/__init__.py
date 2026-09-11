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
"""Save captured CUDA graphs and rebuild them in a fresh process.

Every scheduler process pays the full capture cost at start: two warmup
forwards, a capture and an instantiate per phase and shape bucket, with
cross-rank barriers in between. This package lets one "reference" start
capture normally and every later start rebuild replayable executables from
files, with bit-identical numerics. Saved graphs are templates: no request
state, KV contents or counters are recovered (cold-start speedup only).

Three concerns, three module groups:

``regions`` and ``utils.cuda_vmm_utils.FixedArena`` -- **address space**.
    The artifact never stores a device pointer; every pointer word is a
    ``RegionRef(region_id, offset)`` into an owner-named region. At load a
    ``RegionPlacementPolicy`` gives each region a disposition: ``RELOCATE``
    (patch every word by the region's delta; the correctness baseline, proven
    complete for every kernel class SGLang launches on Blackwell), ``PIN``
    (the region must land at its saved address, asserted by name; the
    fixed-VA accelerator built on ``cuMemAddressReserve`` at a requested
    address, which is deterministic) or ``REJECT`` (the shape recaptures).

``comm`` -- **communicator checkpoint**.
    What a loaded graph needs from each communicator of a group: its
    identity, the custom all-reduce pointer-table rows at their saved
    absolute indices (restored after pre-advancing the row counter so
    recaptured shapes allocate above them), NCCL symmetric-memory windows
    re-registered against the new ``ncclComm_t``, and a loud
    ``needs_recapture`` for any graph that launches an NCCL kernel, because
    the communicator handle inside its parameters is not relocatable.

``format``, ``codec``, ``kernels``, ``safety``, ``materializer``, ``store``
-- **graph structure**.
    ``format`` is the msgspec artifact; ``codec`` encodes a live ``CUgraph``
    into it and materializes it back through ``cuGraphAddKernelNode`` with
    kernels re-resolved by ``(container sha256, name)``; ``safety`` is the
    fail-closed ladder (per-graph verdicts agreed across ranks, self-check,
    shadow verify, strict mode); ``materializer`` is the one seam in the
    runners that decides capture versus import per shape; ``store`` is the
    on-disk layout.

Backends gain two abstract methods (``export_shape`` / ``import_shape``) and
runners route their per-shape capture through ``materialize_shape``. With
``--cuda-graph-cache-mode off`` (the default) behaviour is unchanged.

Status: draft. Interfaces and pure logic are implemented and unit tested;
anything that needs the CUDA driver or a second process raises
``NotImplementedError`` naming the design section that specifies it.
"""

from __future__ import annotations

from sglang.srt.model_executor.graph_serialization.fingerprint import (  # noqa: F401
    GraphArtifactFingerprint,
    diff_fingerprints,
    fingerprint_digest,
)
from sglang.srt.model_executor.graph_serialization.format import (  # noqa: F401
    FORMAT_VERSION,
    ArtifactManifest,
    GraphVerdict,
    RankManifest,
    RegionKind,
    RegionRef,
    RegionSpec,
    RunnerBundle,
    SerializedGraph,
    ShapeArtifact,
    ShapeKeyRecord,
    shape_artifact_verdict,
    unsupported_shape_artifact,
)
from sglang.srt.model_executor.graph_serialization.plan import (  # noqa: F401
    CacheMode,
    GraphSerializationPlan,
    Placement,
    VerifyMode,
    read_plan_from_config,
)
