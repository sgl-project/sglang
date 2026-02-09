# Copyright 2023-2024 SGLang Team
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
"""Ray actor wrapper for SGLang Engine.

This module is imported on the head/driver node which may be
CPU-only. Do NOT add sglang imports at module level. All sglang imports
happen inside _InternalEngineActor.__init__() on GPU worker nodes.

Usage:

    from sglang.srt.entrypoints.engine_actor import EngineActor

    engine = EngineActor(model_path="meta-llama/Llama-2-7b", tp_size=4)
    result = engine.generate(prompt="Hello")
    engine.shutdown()
"""

from __future__ import annotations

import ray


@ray.remote
class _InternalEngineActor:
    """Internal Ray actor that runs sglang.Engine on a GPU worker node."""

    def __init__(self, **engine_kwargs):
        from sglang import Engine

        pgs = engine_kwargs.pop("_ray_placement_groups", None)
        self.engine = Engine(_ray_placement_groups=pgs, **engine_kwargs)

    def is_ready(self) -> bool:
        return self.engine is not None

    def generate(self, **kwargs):
        return self.engine.generate(**kwargs)

    def encode(self, prompt, **kwargs):
        return self.engine.encode(prompt=prompt, **kwargs)

    def get_server_info(self):
        return self.engine.get_server_info()

    def flush_cache(self):
        return self.engine.flush_cache()

    def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None


class EngineActor:
    """Wrapper that manages a Ray actor running sglang.Engine.

    Safe to import and instantiate on CPU-only head nodes.
    Does NOT import sglang - all sglang code runs on GPU workers.
    """

    def __init__(self, **engine_kwargs):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from sglang.srt.utils.ray_utils import create_placement_groups

        pgs = create_placement_groups(
            tp_size=engine_kwargs.get("tp_size", 1),
            pp_size=engine_kwargs.get("pp_size", 1),
            nnodes=engine_kwargs.get("nnodes", 1),
        )

        # Launch internal actor on rank 0's node (co-location for ZMQ IPC)
        self._actor = _InternalEngineActor.options(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pgs[0],
                placement_group_bundle_index=0,
            ),
        ).remote(_ray_placement_groups=pgs, **engine_kwargs)

        # Wait for engine to initialize
        ray.get(self._actor.is_ready.remote())

    def generate(self, **kwargs):
        return ray.get(self._actor.generate.remote(**kwargs))

    def encode(self, prompt, **kwargs):
        return ray.get(self._actor.encode.remote(prompt=prompt, **kwargs))

    def get_server_info(self):
        return ray.get(self._actor.get_server_info.remote())

    def flush_cache(self):
        return ray.get(self._actor.flush_cache.remote())

    def shutdown(self):
        if self._actor is not None:
            ray.get(self._actor.shutdown.remote())
            self._actor = None
