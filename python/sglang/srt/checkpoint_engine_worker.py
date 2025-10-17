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

import zmq
from checkpoint_engine.worker import update_weights_from_ipc

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
