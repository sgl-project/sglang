import logging
import math
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

def _import_wpi_client():
    """Lazily import WPIClient."""
    try:
        from wpi_client.client import WPIClient
        return WPIClient
    except ImportError:
        raise ImportError(
            "WPI weight transfer requires the `wpi_client` package. "
            "Please install it."
        ) from None

class SGLangWPIWorkerExtension:
    """Worker extension for SGLang to support WPI weight updates."""

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.client = None
        self.vram_buffer = None
        self.buffer_id = ""
        self.buffer_size = 0
        self.staged = False

    def init_wpi(self, buffer_id: str, buffer_size: int, socket_dir: str, driver_port: int):
        """Initialize WPI client and map memory."""
        WPIClient = _import_wpi_client()
        self.buffer_id = buffer_id
        self.buffer_size = buffer_size

        self.client = WPIClient(socket_dir=socket_dir, driver_port=driver_port)

        # Assume TP rank/size maps to shard index/total shards
        shard_index = self.model_runner.tp_rank
        total_shards = self.model_runner.tp_size
        if total_shards <= 1:
            shard_index = -1
            total_shards = 0

        self.client.stage_weight(
            buffer_id=self.buffer_id,
            size_bytes=self.buffer_size,
            claim_id=f"{self.buffer_id}-claim",
            shard_index=shard_index,
            total_shards=total_shards,
        )
        self.staged = True

        # Receive FD and map memory
        device_id = torch.cuda.current_device()
        fd = self.client.receive_fd(
            self.buffer_id,
            gpu_id=device_id,
            shard_index=shard_index,
            total_shards=total_shards,
        )
        device_ptr = self.client.import_cuda_memory(fd, self.buffer_size, device_id=device_id)
        self.vram_buffer = self.client.wrap_as_buffer(device_ptr, self.buffer_size)

        # Connect to notify socket
        self.client.connect_notify_socket(
            self.buffer_id,
            shard_index=shard_index,
            total_shards=total_shards,
        )
        logger.info(f"WPI: Worker initialized for buffer {buffer_id}, shard {shard_index}/{total_shards}")

    def update_weights_from_wpi(self, update_info: dict):
        """Wait for READY and load weights from mapped memory."""
        if self.client is None:
            raise RuntimeError("WPI not initialized. Call init_wpi first.")

        logger.info("WPI: Waiting for READY signal...")
        self.client.wait_for_ready(timeout=300.0)

        names = update_info["names"]
        dtype_names = update_info["dtype_names"]
        shapes = update_info["shapes"]
        offsets = update_info["offsets"]

        def weight_iterator():
            for name, dtype_name, shape, offset in zip(names, dtype_names, shapes, offsets):
                dtype = getattr(torch, dtype_name)
                num_elements = math.prod(shape)
                nbytes = num_elements * dtype.itemsize

                weight = (
                    self.vram_buffer[offset:offset + nbytes]
                    .view(dtype=dtype)
                    .view(shape)
                )
                yield name, weight

        logger.info(f"WPI: Loading {len(names)} weights into model...")
        self.model_runner.model.load_weights(weight_iterator())

        # Post hook processing (quantization etc.)
        self._post_hook()
        logger.info("WPI: Weight update completed successfully")

    def _post_hook(self):
        """Perform post-processing after weight loading."""
        try:
            from sglang.srt.model_loader.loader import device_loading_context

            # Process quantization methods after loading weights
            for _, module in self.model_runner.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    target_device = torch.device("cuda", torch.cuda.current_device())
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
            
            # Call model-specific post-loading hook if available
            if hasattr(self.model_runner.model, "post_load_weights"):
                self.model_runner.model.post_load_weights()
        except Exception as e:
            logger.warning(f"WPI: Post-hook processing failed: {e}")

    def shutdown(self):
        """Clean up resources."""
        if self.client is not None:
            if self.staged:
                try:
                    self.client.unstage_weight(f"{self.buffer_id}-claim")
                except Exception as e:
                    logger.warning(f"WPI: Error during unstage: {e}")
                self.staged = False
            self.client.close()
            self.client = None
        self.vram_buffer = None
