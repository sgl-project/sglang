# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs import Lfm2Config, Mamba2Config, MambaConfig
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeterministicCheckpoints(unittest.TestCase):
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
                pool = SimpleNamespace(
                    mamba_ckpt_pool=None,
                    mamba_pool=Mock(replayssm_write_pos=None),
                    translate_mamba_indices=lambda indices: indices,
                )
                component = MambaComponent.__new__(MambaComponent)
                component.cache = SimpleNamespace(
                    enable_mamba_extra_buffer=False, req_to_token_pool=pool
                )
                component._alloc_mamba_slot = Mock(return_value=torch.tensor([7]))
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
                    component._alloc_mamba_slot.assert_not_called()
                    pool.mamba_pool.copy_from.assert_not_called()
                elif finished:
                    torch.testing.assert_close(params.mamba_value, torch.tensor([3]))
                else:
                    torch.testing.assert_close(params.mamba_value, torch.tensor([7]))
                    pool.mamba_pool.copy_from.assert_called_once()


if __name__ == "__main__":
    unittest.main()
