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

"""Dynamic LoRA endpoints vs tokenizer-worker-num (#31084)."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARegistry
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MIXIN = "sglang.srt.managers.tokenizer_control_mixin"


class _FakeTokenizerManager(TokenizerControlMixin):
    def __init__(
        self,
        tokenizer_worker_num: int,
        dp_size: int = 1,
        enable_dp_attention: bool = False,
    ):
        self.server_args = SimpleNamespace(
            enable_lora=True,
            dp_size=dp_size,
            enable_dp_attention=enable_dp_attention,
            tokenizer_worker_num=tokenizer_worker_num,
            max_loaded_loras=None,
        )
        self.auto_create_handle_loop = MagicMock()
        self.lora_update_lock = asyncio.Lock()
        self.lora_registry = LoRARegistry()
        self.lora_ref_cache = {}
        self.update_lora_adapter_communicator = AsyncMock(
            return_value=[LoRAUpdateOutput(success=True, loaded_adapters={})]
        )


def _load_req():
    return LoadLoRAAdapterReqInput(lora_name="adapter_a", lora_path="/tmp/adapter_a")


def _load_tensors_req():
    return LoadLoRAAdapterFromTensorsReqInput(
        lora_name="adapter_a",
        config_dict={"r": 8},
        serialized_named_tensors=[b"tp0-bytes"],
    )


def _unload_req():
    return UnloadLoRAAdapterReqInput(lora_name="adapter_a")


class _LoRAControlTestBase(CustomTestCase):
    def setUp(self):
        self._p_lora = patch(
            f"{_MIXIN}.get_lora",
            return_value=SimpleNamespace(enable_lora=True),
        )
        self._p_parallel = patch(
            f"{_MIXIN}.get_parallel",
            return_value=SimpleNamespace(dp_size=1, enable_dp_attention=False),
        )
        self._p_lora.start()
        self._p_parallel.start()

    def tearDown(self):
        self._p_parallel.stop()
        self._p_lora.stop()

    def _set_parallel(self, dp_size, enable_dp_attention=False):
        self._p_parallel.stop()
        self._p_parallel = patch(
            f"{_MIXIN}.get_parallel",
            return_value=SimpleNamespace(
                dp_size=dp_size, enable_dp_attention=enable_dp_attention
            ),
        )
        self._p_parallel.start()


class TestMultiTokenizerRejectsDynamicLoRA(_LoRAControlTestBase):
    def _expect_guard(self, manager, result):
        self.assertFalse(result.success)
        self.assertIn("--tokenizer-worker-num", result.error_message)
        self.assertIn("31084", result.error_message)
        # Guard must describe both load and unload (same message for all 3 APIs).
        self.assertRegex(result.error_message, r"load(/|ing).*(unload|unloading)")
        manager.update_lora_adapter_communicator.assert_not_awaited()
        self.assertEqual(manager.lora_registry.num_registered_loras, 0)
        self.assertEqual(manager.lora_ref_cache, {})

    def test_load_rejected(self):
        mgr = _FakeTokenizerManager(tokenizer_worker_num=2)
        self._expect_guard(mgr, asyncio.run(mgr.load_lora_adapter(_load_req())))

    def test_load_from_tensors_rejected(self):
        mgr = _FakeTokenizerManager(tokenizer_worker_num=2)
        self._expect_guard(
            mgr,
            asyncio.run(mgr.load_lora_adapter_from_tensors(_load_tensors_req())),
        )

    def test_unload_rejected(self):
        mgr = _FakeTokenizerManager(tokenizer_worker_num=2)
        self._expect_guard(mgr, asyncio.run(mgr.unload_lora_adapter(_unload_req())))

    def test_still_rejected_with_dp_attention(self):
        self._set_parallel(dp_size=2, enable_dp_attention=True)
        mgr = _FakeTokenizerManager(
            tokenizer_worker_num=2, dp_size=2, enable_dp_attention=True
        )
        self._expect_guard(mgr, asyncio.run(mgr.load_lora_adapter(_load_req())))


class TestSingleTokenizerUnchanged(_LoRAControlTestBase):
    def test_load_and_unload(self):
        # Use one event loop: asyncio.Lock must not be reused across asyncio.run().
        async def _run():
            mgr = _FakeTokenizerManager(tokenizer_worker_num=1)
            load = await mgr.load_lora_adapter(_load_req())
            self.assertTrue(load.success)
            mgr.update_lora_adapter_communicator.assert_awaited()
            self.assertEqual(mgr.lora_registry.num_registered_loras, 1)
            self.assertIn("adapter_a", mgr.lora_ref_cache)

            unload = await mgr.unload_lora_adapter(_unload_req())
            self.assertTrue(unload.success)
            self.assertEqual(mgr.lora_registry.num_registered_loras, 0)
            return mgr

        asyncio.run(_run())

    def test_load_from_tensors(self):
        mgr = _FakeTokenizerManager(tokenizer_worker_num=1)
        load = asyncio.run(mgr.load_lora_adapter_from_tensors(_load_tensors_req()))
        self.assertTrue(load.success)
        mgr.update_lora_adapter_communicator.assert_awaited_once()
        self.assertEqual(mgr.lora_registry.num_registered_loras, 1)
        self.assertIn("adapter_a", mgr.lora_ref_cache)


if __name__ == "__main__":
    unittest.main()
