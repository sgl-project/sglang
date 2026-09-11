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

Capturing CUDA graphs is the largest fixed startup cost after weight loading
and is paid on every restart. After one reference start captures normally,
this package lets later starts rebuild replayable executables from files with
bit-identical numerics. Saved graphs are templates: no request state, KV
contents or counters are recovered.

Four modules, one concern each:

``memory``    every pointer word of a saved graph is a ``RegionRef`` into an
              owner-named region; ``MemoryMap.bind`` relocates or pins regions
              at load (VMM ``FixedArena`` reservations for the pinned kinds).
``comm``      what a loaded graph needs from each communicator: identity,
              custom all-reduce rows at absolute indices, NCCL symmetric-memory
              windows; graphs launching NCCL kernels recapture.
``kernels``   kernel identity ``(container sha256, name)`` and the tiers that
              resolve it in a fresh process.
``loadstore`` the artifact, ``save_graph`` / ``load_graph``, the file store
              and ``GraphCache``, the one seam runners talk to.

Backends implement ``export_shape`` / ``import_shape``; runners call
``GraphCache.materialize`` per shape. With ``--cuda-graph-cache-mode off``
(default) nothing changes. This is a draft: interfaces are defined, everything
that needs the CUDA driver raises ``NotImplementedError`` naming its design
section in ``DESIGN_cuda_graph_serialization.md``.

Import names from the submodule that defines them; this package re-exports
nothing.
"""
