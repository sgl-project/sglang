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
    import ray
    ray.init()

    from sglang.srt.entrypoints.engine_actor import create_engine_actor_class

    EngineActor = create_engine_actor_class()
    engine_actor = EngineActor.options(num_gpus=4).remote(
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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def create_engine_actor_class():
    """Factory function to create EngineActor class with Ray decorator.

    Returns a Ray actor class that wraps sglang.Engine. The Engine is
    imported and created INSIDE the actor, which runs on a GPU worker node.
    This is necessary when the head node doesn't have GPU access or when
    sglang imports would fail on the head node.
    """
    import ray

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
            """
            # Import sglang INSIDE the actor (runs on GPU worker node)
            from sglang import Engine

            logger.info(f"Creating Engine with kwargs: {list(engine_kwargs.keys())}")
            self.engine = Engine(**engine_kwargs)
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
            return_logprob: Optional[Union[List[bool], bool]] = False,
            logprob_start_len: Optional[Union[List[int], int]] = None,
            top_logprobs_num: Optional[Union[List[int], int]] = None,
            token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
            lora_path: Optional[List[Optional[str]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Dict, Iterator[Dict]]:
            """Generate text using the engine.

            See sglang.Engine.generate() for full documentation.
            """
            return self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                input_ids=input_ids,
                image_data=image_data,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
                lora_path=lora_path,
                stream=stream,
                **kwargs,
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

    return EngineActor
