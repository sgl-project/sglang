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

This module provides a Ray actor that wraps the SGLang Engine, allowing
sglang to be imported INSIDE the actor (on GPU worker nodes) rather than
on the head node which may not have GPU access.

Usage:
    from sglang.srt.entrypoints.engine_actor import create_and_launch_engine_actor

    engine_actor = create_and_launch_engine_actor(
        model_path="meta-llama/Llama-2-7b-hf",
        tp_size=4,
        use_ray=True,
    )

    # Wait for initialization
    ray.get(engine_actor.is_ready.remote())

    # Generate
    result = ray.get(engine_actor.generate.remote(prompt="Hello"))
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


def create_and_launch_engine_actor(**engine_kwargs):
    """Launch EngineActor on rank 0's worker node for optimal IPC communication.

    This function:
    1. Discovers GPU topology in the Ray cluster
    2. Creates per-node placement groups
    3. Schedules EngineActor on rank 0's node (ensuring IPC works for ZMQ)
    4. Passes pre-computed topology/PGs to Engine to avoid double-creation

    Args:
        **engine_kwargs: All arguments passed to sglang.Engine()
            Common args: model_path, tp_size, pp_size, port, use_ray, etc.

    Returns:
        Ray actor handle for the EngineActor.
    """
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from sglang.srt.utils.ray_cluster_utils import (
        RayClusterInfo,
        create_cluster_topology,
        create_per_node_placement_groups,
        discover_gpu_nodes,
    )

    tp_size = engine_kwargs.get("tp_size", 1)
    pp_size = engine_kwargs.get("pp_size", 1)
    world_size = tp_size * pp_size

    gpu_nodes = discover_gpu_nodes()
    if len(gpu_nodes) == 0:
        raise RuntimeError("No GPU nodes found in Ray cluster")

    topology = create_cluster_topology(gpu_nodes, world_size)
    placement_groups = create_per_node_placement_groups(topology)

    logger.info(
        f"Launching EngineActor on rank 0's node: "
        f"{topology.nnodes} nodes, {topology.gpus_per_node} GPUs/node, "
        f"world_size={world_size}"
    )

    @ray.remote
    class EngineActor:
        """Ray actor wrapper for SGLang Engine.

        The Engine is created inside this actor, which runs on a GPU worker node.
        This ensures sglang imports happen on a node with GPU access.
        """

        def __init__(self, **engine_kwargs):
            """Initialize the Engine inside the actor.

            Args:
                **engine_kwargs: All arguments passed to sglang.Engine()
                    Common args: model_path, tp_size, pp_size, port, use_ray, etc.
                    ray_cluster_info (RayClusterInfo) is passed through to
                    Engine which forwards it to _launch_workers().
            """
            import ray
            from sglang import Engine
            from sglang.srt.utils.ray_cluster_utils import RayClusterInfo

            ray_cluster_info = engine_kwargs.pop("ray_cluster_info", None) or RayClusterInfo()
            # This actor runs on PG[0] (rank 0's node), so local IP = rank 0 IP.
            # Pass it through to avoid a probe actor round-trip (~100-500ms).
            if ray_cluster_info.rank0_node_ip is None:
                ray_cluster_info.rank0_node_ip = ray.util.get_node_ip_address()

            logger.info(f"Creating Engine with kwargs: {list(engine_kwargs.keys())}")
            self.engine = Engine(ray_cluster_info=ray_cluster_info, **engine_kwargs)
            logger.info("Engine created successfully")

        def is_ready(self) -> bool:
            """Check if the engine is ready."""
            return self.engine is not None

        def generate(
            self,
            prompt: Optional[Union[List[str], str]] = None,
            sampling_params: Optional[Union[List[Dict], Dict]] = None,
            input_ids: Optional[Union[List[List[int]], List[int]]] = None,
            image_data: Optional[Any] = None,
            audio_data: Optional[Any] = None,
            video_data: Optional[Any] = None,
            return_logprob: Optional[Union[List[bool], bool]] = False,
            logprob_start_len: Optional[Union[List[int], int]] = None,
            top_logprobs_num: Optional[Union[List[int], int]] = None,
            token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
            lora_path: Optional[List[Optional[str]]] = None,
            custom_logit_processor: Optional[Union[List[str], str]] = None,
            return_hidden_states: bool = False,
            return_routed_experts: bool = False,
            stream: bool = False,
            bootstrap_host: Optional[Union[List[str], str]] = None,
            bootstrap_port: Optional[Union[List[int], int]] = None,
            bootstrap_room: Optional[Union[List[int], int]] = None,
            data_parallel_rank: Optional[int] = None,
            external_trace_header: Optional[Dict] = None,
            rid: Optional[Union[List[str], str]] = None,
        ) -> Union[Dict, Iterator[Dict]]:
            """Generate text using the engine.

            See sglang.Engine.generate() for full documentation.
            """
            return self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                input_ids=input_ids,
                image_data=image_data,
                audio_data=audio_data,
                video_data=video_data,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
                lora_path=lora_path,
                custom_logit_processor=custom_logit_processor,
                return_hidden_states=return_hidden_states,
                return_routed_experts=return_routed_experts,
                stream=stream,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
                data_parallel_rank=data_parallel_rank,
                external_trace_header=external_trace_header,
                rid=rid,
            )

        def encode(
            self,
            prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
            **kwargs,
        ) -> Dict:
            """Encode text to embeddings.

            See sglang.Engine.encode() for full documentation.
            """
            return self.engine.encode(prompt=prompt, **kwargs)

        def get_server_info(self) -> Dict[str, Any]:
            """Get server information."""
            return self.engine.get_server_info()

        def flush_cache(self):
            """Flush the KV cache."""
            return self.engine.flush_cache()

        def shutdown(self):
            """Shutdown the engine."""
            if self.engine is not None:
                self.engine.shutdown()
                self.engine = None

    engine_actor = EngineActor.options(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_groups[0],
            placement_group_bundle_index=0,
        ),
    ).remote(
        ray_cluster_info=RayClusterInfo(
            topology=topology,
            placement_groups=placement_groups,
        ),
        **engine_kwargs,
    )

    return engine_actor
