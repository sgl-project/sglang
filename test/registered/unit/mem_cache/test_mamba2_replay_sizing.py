import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
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


class TestMamba2ReplaySizing(CustomTestCase):
    def test_pointer_table_layer_views_and_contiguous_columns(self):
        # FlashInfer dereferences contiguous int64 columns; pool State slices
        # axis zero by layer. A row-major table would silently violate its ABI.
        from sglang.kernels.ops.mamba.flashinfer_replay_materialize import (
            make_replay_pointer_table,
        )

        tensors = [
            torch.empty(shape)
            for shape in (
                (2, 3, 4, 64, 128),
                (2, 3, 4, 8, 64),
                (2, 3, 1, 8, 128),
                (2, 3, 4, 8),
                (2, 4),
            )
        ]
        table = make_replay_pointer_table(*tensors)
        self.assertEqual(table.shape, (2, 11))
        for column in table.unbind(1):
            self.assertTrue(column.is_contiguous())
        for layer in range(2):
            for family, tensor in enumerate(tensors[:4]):
                self.assertEqual(
                    table[layer, 2 * family].item(), tensor[layer].data_ptr()
                )
                self.assertEqual(table[layer, 2 * family + 1].item(), tensor.stride(1))
            self.assertEqual(table[layer, 8].item(), tensors[4][layer].data_ptr())
            self.assertEqual(table[layer, 9:].count_nonzero().item(), 0)

    def test_configurator_auto_explicit_and_no_radix(self):
        """Sizing must use the configurator's captured DP width, not runner fields."""
        from sglang.srt import runtime_context as rc
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        params = mamba2_params()
        costs = Mamba2ReplaySizing.from_params(
            params, layers=40, width=4, activation_bytes=2
        )
        # Use the real slotted type so a removed runner attribute cannot be
        # accidentally supplied by a permissive test double. Skip model setup;
        # this exercises only the CPU memory-budget calculation.
        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        configurator.mambaish_config = SimpleNamespace(mamba2_cache_params=params)
        configurator.pp_size = 1
        configurator.model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            num_hidden_layers=88,
            hf_text_config=SimpleNamespace(model_type="nemotron_h"),
        )
        configurator.spec_algorithm = SpeculativeAlgorithm.EAGLE
        configurator.is_draft_worker = False
        for fixed, no_radix, cap, dp in (
            (None, False, 256, 1),
            (238, False, 256, 1),
            (None, True, 32, 1),
            (None, False, 256, 2),
            (238, False, 256, 2),
            (None, True, 32, 2),
        ):
            configurator.attn_dp_size = dp
            with (
                self.subTest(fixed=fixed, no_radix=no_radix, dp=dp),
                rc.get_context().override_server_args(
                    enable_linear_replayssm_spec=True,
                    max_mamba_cache_size=fixed,
                    disable_radix_cache=no_radix,
                    max_running_requests=cap,
                    speculative_num_draft_tokens=4,
                    mamba_full_memory_ratio=0.9,
                    mamba_radix_cache_strategy="extra_buffer",
                ),
            ):
                ratio = configurator._calculate_mamba_ratio()
                request_cap = cap // dp
                remaining = configurator._handle_max_mamba_cache(74.0)
                slots = rc.get_schedule().max_mamba_cache_size
                expected = (
                    fixed // dp
                    if fixed is not None
                    else request_cap
                    if no_radix
                    else costs.solve(74 * (1 << 30) * 0.9 / 1.9, request_cap, ratio)
                )
                self.assertEqual(slots, expected)
                self.assertEqual(
                    remaining,
                    74 - costs.bytes_for(slots, request_cap, ratio) / (1 << 30),
                )

    def test_actual_geometry_and_exact_solve(self):
        costs = Mamba2ReplaySizing.from_params(
            mamba2_params(), layers=40, width=4, activation_bytes=2
        )
        self.assertEqual(costs.persistent_per_slot, 86343680)
        self.assertEqual(costs.record_per_slot, 6062400)
        self.assertEqual(costs.conv_per_row, 4915200)
        self.assertEqual(costs.parameters, 24320)
        for cap in (16, 47, 256):
            for ratio in (1, 4, 5):
                for budget in (1 << 30, 20 << 30, 35 << 30):
                    slots = costs.solve(budget, cap, ratio)
                    self.assertLessEqual(costs.bytes_for(slots, cap, ratio), budget)
                    self.assertGreater(costs.bytes_for(slots + 1, cap, ratio), budget)

    def test_gates(self):
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.server_args import ServerArgs

        valid = dict(
            mamba_backend="flashinfer",
            mamba_ssm_dtype="float16",
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            disaggregation_mode="null",
            enable_unified_memory=False,
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=True,
        )
        validate_mamba2_spec_replay(
            resolving_view(ServerArgs(model_path="dummy", **valid)),
            "nemotron_h",
            is_cuda=True,
            resolved=True,
        )
        for change in (
            dict(speculative_eagle_topk=2),
            dict(speculative_algorithm="DSPARK"),
            dict(speculative_num_draft_tokens=17),
            dict(speculative_num_draft_tokens=None),
            dict(disaggregation_mode="decode"),
            dict(enable_unified_memory=True),
            dict(enable_linear_replayssm=True),
            dict(mamba_backend="triton"),
            dict(mamba_ssm_dtype="bfloat16"),
            dict(enable_int8_mamba_checkpoint=True),
            dict(enable_page_major_kv_layout=True),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_mamba2_spec_replay(
                    resolving_view(ServerArgs(model_path="dummy", **(valid | change))),
                    "nemotron_h",
                    is_cuda=True,
                    resolved=True,
                )
        for model, cuda in (("qwen3_next", True), ("nemotron_h", False)):
            with self.assertRaises(ValueError):
                validate_mamba2_spec_replay(
                    resolving_view(ServerArgs(model_path="dummy", **valid)),
                    model,
                    is_cuda=cuda,
                    resolved=True,
                )

    def test_global_flag_dispatch_excludes_drafts_and_other_families(self):
        """A shared switch must not allocate Mamba2 scratch for GDN/KDA or drafts."""
        from sglang.srt import runtime_context as rc
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        for enabled in (False, True):
            with rc.get_context().override_server_args(
                enable_linear_replayssm_spec=enabled
            ):
                for model_type in ("nemotron_h", "qwen3_next", "kimi_linear", "other"):
                    configurator.model_config = SimpleNamespace(
                        hf_text_config=SimpleNamespace(model_type=model_type)
                    )
                    for draft in (False, True):
                        configurator.is_draft_worker = draft
                        with self.subTest(
                            enabled=enabled, model=model_type, draft=draft
                        ):
                            self.assertEqual(
                                configurator._mamba2_spec_replay_enabled,
                                enabled and model_type == "nemotron_h" and not draft,
                            )

    def test_global_flag_preserves_model_specific_resolution(self):
        """Mamba2 must not inherit GDN/KDA's FP32 default or replay backend gate."""
        from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
        from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.runtime_context import override_platform
        from sglang.srt.server_args import ServerArgs

        for model_type in ("nemotron_h", "qwen3_next", "kimi_linear"):
            dtype = "float16" if model_type == "nemotron_h" else None
            args = ServerArgs(
                model_path="dummy",
                enable_linear_replayssm_spec=True,
                mamba_ssm_dtype=dtype,
                mamba_backend="flashinfer" if model_type == "nemotron_h" else "triton",
                speculative_algorithm="EAGLE",
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
            )
            # Model metadata is an external loading boundary; no weights/download.
            model = SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type=model_type)
            )
            with (
                patch.object(args, "_model_config", model, create=True),
                override_platform(is_cuda=True, is_sm100=True, has_flashinfer=True),
            ):
                handle_mamba_backend(args)
                handle_linear_attn_backend(args)
            self.assertEqual(
                resolution_result(args, "mamba_ssm_dtype"),
                "float16" if model_type == "nemotron_h" else "float32",
            )

    def test_cli_exposes_only_shared_replay_switch(self):
        import argparse
        import contextlib
        import io

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            ["--model", "dummy", "--enable-linear-replayssm-spec"]
        )
        self.assertTrue(parsed.enable_linear_replayssm_spec)
        self.assertFalse(hasattr(parsed, "enable_mamba2_spec_replay"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--model", "dummy", "--enable-mamba2-spec-replay"])


if __name__ == "__main__":
    unittest.main()
