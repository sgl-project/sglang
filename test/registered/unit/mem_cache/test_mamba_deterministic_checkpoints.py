# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.configs.mamba import MambaConfig
from sglang.srt.configs.mamba2 import Mamba2Config
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeterministicCheckpoints(CustomTestCase):
    def test_no_buffer_only_donates_aligned_prefill_states(self):
        for config, deterministic, finished, length, expected in (
            (Mamba2Config(), True, False, 255, 0),
            (Mamba2Config(), True, False, 256, 256),
            (Mamba2Config(), True, False, 512, 512),
            (Mamba2Config(), True, True, 256, 0),
            (Mamba2Config(), True, True, 513, 0),
            (Mamba2Config(), False, True, 513, 513),
            (MambaConfig(), True, True, 513, 513),
            (Lfm2Config(), True, True, 513, 513),
        ):
            with self.subTest(
                model=type(config).__name__,
                deterministic=deterministic,
                finished=finished,
                length=length,
            ):
                storage = SimpleNamespace(
                    tree_sub_pool=lambda name: "mamba",
                    quantizes_checkpoints=lambda sub_pool: False,
                    settle_replay=Mock(return_value=0),
                    copy_state=Mock(),
                )
                component = MambaComponent.__new__(MambaComponent)
                component.host_only_checkpoint = False
                component.cache = SimpleNamespace(
                    enable_mamba_extra_buffer=False, kv_storage=lambda: storage
                )
                component._try_alloc_mamba_slot = Mock(return_value=torch.tensor([7]))
                req = SimpleNamespace(kv=ReqKvInfo(mamba_pool_idx=torch.tensor(3)))
                params = InsertParams()
                execution = SimpleNamespace(
                    deterministic=SimpleNamespace(
                        enable_deterministic_inference=deterministic
                    )
                )
                with (
                    patch(
                        "sglang.srt.mem_cache.unified_cache.components.mamba.get_exec",
                        return_value=execution,
                    ),
                    patch(
                        "sglang.srt.mem_cache.unified_cache.components.mamba.process_model_config",
                        return_value=SimpleNamespace(hf_config=config),
                    ),
                ):
                    actual = component.prepare_for_caching_req(
                        req, params, length, finished
                    )
                self.assertEqual(actual, expected)
                if expected == 0:
                    self.assertIsNone(params.mamba_value)
                    component._try_alloc_mamba_slot.assert_not_called()
                    storage.copy_state.assert_not_called()
                elif finished:
                    torch.testing.assert_close(params.mamba_value, torch.tensor([3]))
                else:
                    torch.testing.assert_close(params.mamba_value, torch.tensor([7]))
                    storage.copy_state.assert_called_once_with(
                        "mamba", torch.tensor([3]), torch.tensor([7])
                    )


if __name__ == "__main__":
    unittest.main()
