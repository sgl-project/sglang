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
"""
Checkpoint-engine integration for SGLang.
This module provides weight update functionality via IPC for checkpoint-engine compatibility.
"""
import logging
from typing import Callable, Dict, Optional

import torch
import zmq

try:
    from checkpoint_engine.worker import update_weights_from_ipc
except ImportError:
    raise ImportError(
        "checkpoint-engine is not installed. "
        "Please install it with: pip install sglang[checkpoint-engine]"
    )

logger = logging.getLogger(__name__)


class SGLangCheckpointEngineWorkerExtension:
    """
    Worker extension for SGLang to support checkpoint-engine IPC weight updates.
    This class provides the interface needed for checkpoint-engine integration.
    """

    def __init__(self):
        self._zmq_ctx: Optional[zmq.Context] = None

    def get_device_uuid(self) -> str:
        """Get the UUID of current device."""
        # We need to implement this to get the device UUID
        # This will be overridden when integrated into SGLang's worker
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_device_id(self) -> int:
        """Get the device ID."""
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_model_loader(self) -> Callable:
        """Get the model weight loader function."""
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_post_hook(self) -> Optional[Callable]:
        """Get the post-processing hook after weight loading."""
        return None

    def update_weights_from_ipc(self, zmq_handles: Dict[str, str]):
        """
        Update weights from IPC communication.
        Args:
            zmq_handles: Dict mapping device UUID to ZMQ socket path
        """
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        device_uuid = self.get_device_uuid()
        device_id = self.get_device_id()
        if device_uuid not in zmq_handles:
            raise ValueError(
                f"Device UUID {device_uuid} not found in zmq_handles: {list(zmq_handles.keys())}"
            )
        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[device_uuid],
            device_id=device_id,
            run=self.get_model_loader(),
            post_hook=self.get_post_hook(),
        )


class SGLangCheckpointEngineWorkerExtensionImpl(SGLangCheckpointEngineWorkerExtension):
    """
    Implementation of SGLangCheckpointEngineWorkerExtension that integrates with SGLang's model runner.
    This class provides the concrete implementation for checkpoint-engine IPC weight updates.
    """

    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner

    def get_device_uuid(self) -> str:
        """Get the UUID of current device."""
        # Get device UUID for current device
        device_id = torch.cuda.current_device()
        try:
            return f"GPU-{torch.cuda.get_device_properties(device_id).uuid!s}"
        except AssertionError as e:
            raise ValueError(f"Failed to get GPU UUID for device {device_id}") from e

    def get_device_id(self) -> int:
        """Get the device ID."""
        return torch.cuda.current_device()

    def get_model_loader(self) -> Callable:
        """Get the model weight loader function."""
        return self.model_runner.model.load_weights

    def get_post_hook(self) -> Optional[Callable]:
        """Get the post-processing hook after weight loading."""

        def post_hook():
            # Perform post-processing after weight loading similar to DefaultModelLoader
            try:
                from sglang.srt.model_loader.loader import device_loading_context

                # Process quantization methods after loading weights
                for _, module in self.model_runner.model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        # Move parameters to device if needed for quantization processing
                        target_device = torch.device(
                            "cuda", torch.cuda.current_device()
                        )
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)
                # Call model-specific post-loading hook if available
                if hasattr(self.model_runner.model, "post_load_weights"):
                    self.model_runner.model.post_load_weights()
            except Exception as e:
                logger.warning(f"Post-hook processing failed: {e}")

        return post_hook
