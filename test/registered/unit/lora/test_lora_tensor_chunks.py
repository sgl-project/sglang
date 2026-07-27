from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager():
    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(vocab_size=128)
    manager.base_model = MagicMock()
    manager.load_config = MagicMock()
    manager.lora_backend = MagicMock()
    manager.pending_tensor_loras = {}
    manager.configs = {}
    manager.loras = {}
    manager.lora_refs = {}
    manager.num_pinned_loras = 0
    manager.validate_new_adapter = MagicMock()
    return manager


def test_lora_tensor_chunks_register_only_after_final_chunk():
    manager = _manager()
    lora_ref = LoRARef(
        lora_id="adapter-id",
        lora_name="adapter",
        lora_path="__tensor__",
        pinned=False,
    )
    config_dict = {"r": 32}
    config = MagicMock()
    adapter = MagicMock()

    with (
        patch(
            "sglang.srt.lora.lora_manager.LoRAConfig.from_dict",
            return_value=config,
        ),
        patch("sglang.srt.lora.lora_manager.LoRAAdapter", return_value=adapter),
    ):
        first = manager.load_lora_adapter_from_tensors(
            lora_ref,
            {"first": MagicMock()},
            config_dict,
            is_first_chunk=True,
            is_last_chunk=False,
        )
        assert first.success
        assert manager.loras == {}
        assert "adapter-id" in manager.pending_tensor_loras

        last = manager.load_lora_adapter_from_tensors(
            lora_ref,
            {"last": MagicMock()},
            config_dict,
            is_first_chunk=False,
            is_last_chunk=True,
        )

    assert last.success
    assert manager.pending_tensor_loras == {}
    assert manager.loras == {"adapter-id": adapter}
    assert manager.configs == {"adapter-id": config}
    assert manager.lora_refs == {"adapter-id": lora_ref}
    assert adapter.load_weights_from_tensors_chunk.call_count == 2
    adapter.finalize_weights_from_tensors.assert_called_once_with()


def test_lora_tensor_chunk_requires_a_pending_transfer():
    manager = _manager()
    lora_ref = LoRARef(
        lora_id="adapter-id",
        lora_name="adapter",
        lora_path="__tensor__",
        pinned=False,
    )

    result = manager.load_lora_adapter_from_tensors(
        lora_ref,
        {},
        {"r": 32},
        is_first_chunk=False,
        is_last_chunk=True,
    )

    assert not result.success
    assert "no pending tensor transfer" in result.error_message
