import unittest
from types import SimpleNamespace

import torch
from sglang.srt.configs.mamba2_spec_replay import (
    Mamba2ReplaySizing,
    validate_mamba2_spec_replay,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)


def mamba2_params():
    return Mamba2CacheParams(
        shape=Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=8192,
            n_groups=8,
            num_heads=128,
            head_dim=64,
            state_size=128,
            conv_kernel=4,
        ),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float16),
        layers=list(range(40)),
    )


class TestMamba2ReplaySizing(unittest.TestCase):
    def test_configurator_auto_explicit_and_no_radix(self):
        from sglang.srt import runtime_context as rc
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        params = mamba2_params()
        costs = Mamba2ReplaySizing.from_params(
            params, layers=40, width=4, activation_bytes=2
        )
        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(mamba2_cache_params=params),
            ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
            model_config=SimpleNamespace(dtype=torch.bfloat16, num_hidden_layers=88),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            is_draft_worker=False,
            _calculate_mamba_ratio=lambda: 5,
        )
        for fixed, no_radix, cap in (
            (None, False, 256),
            (238, False, 256),
            (None, True, 32),
        ):
            fake._calculate_mamba_ratio = lambda: 1 if no_radix else 5
            with rc.get_context().override_server_args(
                enable_mamba2_spec_replay=True,
                max_mamba_cache_size=fixed,
                disable_radix_cache=no_radix,
                max_running_requests=cap,
                speculative_num_draft_tokens=4,
                mamba_full_memory_ratio=0.9,
            ):
                remaining = KVCacheConfigurator._handle_max_mamba_cache(fake, 74.0)
                slots = rc.get_schedule().max_mamba_cache_size
                expected = (
                    fixed
                    if fixed is not None
                    else cap
                    if no_radix
                    else costs.solve(74 * (1 << 30) * 0.9 / 1.9, cap, 5)
                )
                self.assertEqual(slots, expected)
                self.assertEqual(
                    remaining,
                    74 - costs.bytes_for(slots, cap, 1 if no_radix else 5) / (1 << 30),
                )

    def test_actual_geometry_and_exact_solve(self):
        costs = Mamba2ReplaySizing.from_params(
            mamba2_params(), layers=40, width=4, activation_bytes=2
        )
        self.assertEqual(costs.persistent_per_slot, 86343680)
        self.assertEqual(costs.record_per_slot, 3604800)
        self.assertEqual(costs.conv_per_row, 4915200)
        self.assertEqual(costs.parameters, 320)
        for cap in (16, 47, 256):
            for ratio in (1, 4, 5):
                for budget in (1 << 30, 20 << 30, 35 << 30):
                    slots = costs.solve(budget, cap, ratio)
                    self.assertLessEqual(costs.bytes_for(slots, cap, ratio), budget)
                    self.assertGreater(costs.bytes_for(slots + 1, cap, ratio), budget)

    def test_gates(self):
        valid = dict(
            enable_mamba2_spec_replay=True,
            mamba_backend="flashinfer",
            mamba_ssm_dtype="float16",
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            disaggregation_mode="null",
            enable_unified_memory=False,
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=False,
        )
        validate_mamba2_spec_replay(
            SimpleNamespace(**valid), "nemotron_h", is_cuda=True, resolved=True
        )
        for change in (
            dict(speculative_eagle_topk=2),
            dict(speculative_algorithm="DSPARK"),
            dict(speculative_num_draft_tokens=17),
            dict(speculative_num_draft_tokens=None),
            dict(disaggregation_mode="decode"),
            dict(enable_unified_memory=True),
            dict(enable_linear_replayssm_spec=True),
            dict(mamba_backend="triton"),
            dict(mamba_ssm_dtype="bfloat16"),
            dict(enable_int8_mamba_checkpoint=True),
            dict(enable_page_major_kv_layout=True),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_mamba2_spec_replay(
                    SimpleNamespace(**(valid | change)),
                    "nemotron_h",
                    is_cuda=True,
                    resolved=True,
                )
        for model, cuda in (("qwen3_next", True), ("nemotron_h", False)):
            with self.assertRaises(ValueError):
                validate_mamba2_spec_replay(
                    SimpleNamespace(**valid), model, is_cuda=cuda, resolved=True
                )


if __name__ == "__main__":
    unittest.main()
