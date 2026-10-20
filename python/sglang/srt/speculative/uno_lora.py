"""Loading for UNO's draft LoRA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.runtime_context import get_spec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


# This name is internal to the model-execution process. It is never exposed as
# a request-selectable serving adapter.
_UNO_INTERNAL_LORA_NAME = "__uno_draft__"

# LoRA pool capacity includes the base-model slot.
_UNO_LORA_POOL_CAPACITY = 2
logger = logging.getLogger(__name__)


def uno_lora_backend_for_device(device: str) -> str:
    """Select the internal UNO LoRA backend for the execution device."""
    return "uno_ascend" if str(device).startswith("npu") else "uno_cublas"


def init_uno_lora_manager(
    model_runner: ModelRunner,
) -> tuple[LoRAManager, str]:
    """Load and pin the single UNO draft adapter."""

    if mambaish_config(model_runner.model_config) is not None:
        # UNO writes a LoRA draft followed by a base-only verify into the same
        # model. Recurrent states need separate draft state and accepted-prefix
        # commit/rollback, which the UNO worker does not implement.
        raise ValueError(
            "UNO does not support recurrent or hybrid linear-attention models "
            "(Mamba/GDN/KDA); draft-state isolation and accepted-state commit "
            "are not implemented."
        )

    lora_path = get_spec().uno_lora_path

    uno_ref = LoRARef(
        lora_id=LoRARef.deterministic_id(
            _UNO_INTERNAL_LORA_NAME,
            lora_path,
        ),
        lora_name=_UNO_INTERNAL_LORA_NAME,
        lora_path=lora_path,
        pinned=True,
    )

    manager = LoRAManager(
        base_model=model_runner.model,
        base_hf_config=model_runner.model_config.hf_config,
        max_loras_per_batch=_UNO_LORA_POOL_CAPACITY,
        load_config=model_runner.load_config,
        dtype=model_runner.dtype,
        server_args=model_runner.server_args,
        lora_backend=uno_lora_backend_for_device(model_runner.device),
        # Infer these from the one trained adapter.
        max_lora_rank=None,
        target_modules=None,
        lora_paths=[uno_ref],
    )

    # LoRAManager construction initially makes only the base slot resident.
    # UNO always needs both fixed choices resident:
    #
    #   None             -> base model
    #   uno_ref.lora_id  -> base model + UNO draft LoRA
    manager.fetch_new_loras({None, uno_ref.lora_id})
    logger.info(
        "UNO LoRA initialized: backend=%s, rank=%s, adapter=%s",
        manager.lora_backend.name,
        manager.max_lora_rank,
        lora_path,
    )

    return manager, uno_ref.lora_id


class UnoCudaGraphLoRAState:
    """Retained token-row LoRA routing for UNO draft graph buckets."""

    def __init__(
        self,
        manager: LoRAManager,
        uno_lora_id: str,
        forward_width: int,
    ):
        self.manager = manager
        self.uno_lora_id = uno_lora_id
        self.forward_width = forward_width
        self._draft_batch_infos = {}

    def capture_draft(self, batch_size: int) -> None:
        if batch_size not in self._draft_batch_infos:
            self.manager.prepare_lora_token_segments(
                lora_ids=[None, self.uno_lora_id] * batch_size,
                segment_lens=[1, self.forward_width - 1] * batch_size,
            )
            batch_info = self.manager.lora_backend.batch_info
            batch_info.use_cuda_graph = True
            self._draft_batch_infos[batch_size] = batch_info

        self.activate_draft(batch_size)

    def activate_draft(self, batch_size: int) -> None:
        self.manager.reset_lora_batch()
        self.manager.lora_backend.batch_info = self._draft_batch_infos[batch_size]

    def reset(self) -> None:
        self.manager.reset_lora_batch()
