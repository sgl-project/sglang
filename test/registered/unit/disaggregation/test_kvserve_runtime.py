"""CPU tests for optional KVServe startup and early reservation."""

import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import kvserve_runtime as module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _FakeCompressionConfig:
    """Subset of the optional KVServe config contract used at this boundary."""

    enabled: bool = False
    transport: str = "compression"
    transformer: str = "hadamard"
    seed: int = 3237919422
    split_type: str = "layer"
    score_path: str | None = None
    hybrid_ratio: float = 0.8
    axis_key: str = "channel"
    axis_value: str = "token"
    high_key_num_levels: int = 12
    high_value_num_levels: int = 8
    low_key_num_levels: int = 6
    low_value_num_levels: int = 4
    codec: str = "ans"
    slot_count: int = 2
    logical_blocks_per_chunk: int = 1
    pipeline: tuple[str, ...] = ("transformer", "quantizer", "codec")

    @classmethod
    def from_extra_config(cls, extra):
        values = dict(extra["compression"])
        enabled = values.pop("enable-compression-pipeline", False)
        if not enabled:
            return cls(transport="raw")
        if "pipeline" in values:
            values["pipeline"] = tuple(values["pipeline"])
        return cls(enabled=True, **values)

    @property
    def uses_aggregate(self):
        return "aggregate" in self.pipeline

    @property
    def uses_quantizer(self):
        return not self.uses_aggregate and "quantizer" in self.pipeline

    @property
    def uses_codec(self):
        return not self.uses_aggregate and "codec" in self.pipeline


def _fake_runtime_api():
    api = types.ModuleType("kvserve_v1.compression.runtime")
    api.CompressionConfig = _FakeCompressionConfig
    return api


class TestKVServeStartup(CustomTestCase):
    def config(self, value):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "compression.json"
        path.write_text(json.dumps(value))
        return str(path)

    def test_missing_and_disabled_config_do_not_read_context_or_import_kvserve(self):
        disabled = self.config({"enable-compression-pipeline": False})
        for env in ({}, {"SGLANG_KVSERVE_CONFIG": disabled}):
            with self.subTest(env=env):
                # Neither runtime configuration nor the optional codec is needed
                # when compression is disabled, including non-P/D deployments.
                unavailable = {"sglang.srt.runtime_context": None, "kvserve_v1": None}
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict(sys.modules, unavailable),
                ):
                    self.assertFalse(module.is_enabled())
                    module.prepare_runtime(object())

    def test_invalid_flag_is_rejected(self):
        env = {
            "SGLANG_KVSERVE_CONFIG": self.config(
                {"enable-compression-pipeline": "false"}
            )
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                module.is_enabled()

    def test_unsupported_role_or_backend_fails_before_codec_import(self):
        cases = (
            ("null", "mooncake", "prefill or decode"),
            (None, "mooncake", "prefill or decode"),
            ("prefill", "nixl", "[Mm]ooncake"),
            ("decode", "fake", "[Mm]ooncake"),
        )
        env = {
            "SGLANG_KVSERVE_CONFIG": self.config({"enable-compression-pipeline": True})
        }
        for role, backend, message in cases:
            with self.subTest(role=role, backend=backend):
                context = types.ModuleType("sglang.srt.runtime_context")
                context.get_disagg = lambda: types.SimpleNamespace(
                    disaggregation_mode=role, disaggregation_transfer_backend=backend
                )
                context.get_parallel = Mock(
                    side_effect=AssertionError("unexpected reservation")
                )
                modules = {context.__name__: context, "kvserve_v1": None}
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict(sys.modules, modules),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        module.prepare_runtime(object())
                context.get_parallel.assert_not_called()

    def test_default_profile_and_independent_slot_and_chunk_settings(self):
        api = _fake_runtime_api()
        with patch.dict(sys.modules, {api.__name__: api}):
            default = module.parse_compression_config(
                {"enable-compression-pipeline": True}
            )
            self.assertEqual(default.logical_blocks_per_chunk, 1024)
            self.assertEqual(default.pipeline, ("transformer", "quantizer", "codec"))
            for slots, chunk in ((2, 512), (8, 64)):
                with self.subTest(slots=slots, chunk=chunk):
                    values = {
                        "enable-compression-pipeline": True,
                        "slot_count": slots,
                        "logical_blocks_per_chunk": chunk,
                    }
                    selected = module.parse_compression_config(values)
                    self.assertEqual(
                        (selected.slot_count, selected.logical_blocks_per_chunk),
                        (slots, chunk),
                    )
                    self.assertEqual(
                        module.select_execution_backend(selected, "auto"), "auto"
                    )
                    self.assertEqual(
                        module.select_execution_backend(selected, "fused"), "fused"
                    )
                    self.assertEqual(values["slot_count"], slots)
                    self.assertEqual(values["logical_blocks_per_chunk"], chunk)

    def test_nondefault_stages_and_quantization_select_standard(self):
        cases = (
            {"pipeline": ["codec"]},
            {"pipeline": ["quantizer"]},
            {"pipeline": ["transformer", "codec"]},
            {"pipeline": ["transformer", "quantizer"]},
            {"pipeline": ["quantizer", "codec"]},
            {"axis_key": "token"},
            {"axis_value": "channel"},
            {"high_key_num_levels": 16},
            {
                "split_type": "head",
                "score_path": "head-scores.json",
                "hybrid_ratio": 0.5,
            },
        )
        api = _fake_runtime_api()
        with patch.dict(sys.modules, {api.__name__: api}):
            for changes in cases:
                with self.subTest(changes=changes):
                    selected = module.parse_compression_config(
                        {"enable-compression-pipeline": True, **changes}
                    )
                    self.assertEqual(
                        module.select_execution_backend(selected, "auto"), "standard"
                    )
                    self.assertEqual(
                        module.select_execution_backend(selected, "standard"),
                        "standard",
                    )
                    with self.assertRaisesRegex(ValueError, "default codec profile"):
                        module.select_execution_backend(selected, "fused")

    def test_rejects_noncompressing_or_unsupported_profiles(self):
        cases = (
            ({"pipeline": []}, "quantizer or codec"),
            ({"pipeline": ["transformer"]}, "quantizer or codec"),
            ({"pipeline": ["aggregate"]}, "quantizer or codec"),
            ({"pipeline": ["quantizer"], "axis_key": "tensor"}, "tensor axis"),
            ({"pipeline": ["quantizer"], "axis_value": "tensor"}, "tensor axis"),
            ({"pipeline": ["quantizer"], "split_type": "head"}, "score_path"),
            ({"score_path": "unused.json"}, "score_path"),
            ({"hybrid_ratio": 0.5}, "hybrid_ratio"),
            ({"slot_count": 3}, "bounded slots"),
        )
        api = _fake_runtime_api()
        with patch.dict(sys.modules, {api.__name__: api}):
            for changes, message in cases:
                with self.subTest(changes=changes):
                    with self.assertRaisesRegex(ValueError, message):
                        module.parse_compression_config(
                            {"enable-compression-pipeline": True, **changes}
                        )
            # Quantizer-only validation does not reject an unused axis in a
            # codec-only pipeline; the C++ pipeline has the same rule.
            codec_only = module.parse_compression_config(
                {
                    "enable-compression-pipeline": True,
                    "pipeline": ["codec"],
                    "axis_key": "tensor",
                }
            )
            self.assertEqual(codec_only.pipeline, ("codec",))

    def test_role_and_dimensions_come_from_runtime_context(self):
        for role, slots, banks, execution, changes, expected_backend in (
            ("prefill", 2, 4, "encode", {}, "auto"),
            ("decode", 8, 0, "decode", {}, "auto"),
            ("prefill", 2, 4, "encode", {"pipeline": ["codec"]}, "standard"),
            ("decode", 8, 0, "decode", {"high_key_num_levels": 16}, "standard"),
        ):
            with self.subTest(role=role, changes=changes):
                calls = []
                parallel = types.SimpleNamespace(attn_tp_size=1, attn_dcp_size=1)
                context = types.ModuleType("sglang.srt.runtime_context")
                context.get_parallel = lambda: parallel
                context.get_disagg = lambda: types.SimpleNamespace(
                    disaggregation_mode=role, disaggregation_transfer_backend="mooncake"
                )
                api = _fake_runtime_api()
                api.CacheSpec = lambda *args: args
                plan = types.SimpleNamespace(reserved_bytes=99, retained_driver_bytes=1)
                api.build_memory_plan = Mock(return_value=plan)
                fused = types.ModuleType("kvserve_v1.compression.runtime.fused_runtime")

                def create(*args, **kwargs):
                    calls.append((args, kwargs))
                    return types.SimpleNamespace(
                        plan=plan, bulk=types.SimpleNamespace(numel=lambda: 64)
                    )

                fused.create_runtime = create
                model = types.SimpleNamespace(
                    get_num_kv_heads=lambda tp: 8,
                    head_dim=128,
                    v_head_dim=128,
                    num_hidden_layers=32,
                )
                kvc = types.SimpleNamespace(
                    is_draft_worker=False,
                    use_mla_backend=False,
                    is_hybrid_swa=False,
                    mambaish_config=None,
                    pp_size=1,
                    page_size=16,
                    kv_cache_dtype=module.torch.bfloat16,
                    gpu_id=0,
                    model_config=model,
                    layer_info=types.SimpleNamespace(start_layer=0, end_layer=32),
                )
                env = {
                    "SGLANG_KVSERVE_CONFIG": self.config(
                        {
                            "enable-compression-pipeline": True,
                            "slot_count": slots,
                            **changes,
                        }
                    ),
                    "SGLANG_KVSERVE_EXECUTION_BACKEND": "auto",
                    "KVSERVE_COMPRESSION_BACKEND": "python",
                }
                modules = {
                    context.__name__: context,
                    api.__name__: api,
                    fused.__name__: fused,
                }
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict(sys.modules, modules),
                    patch.dict(module._runtimes, {}, clear=True),
                ):
                    before = os.environ.copy()
                    module.prepare_runtime(kvc)
                    self.assertEqual(os.environ, before)
                selected = api.build_memory_plan.call_args.args[0]
                self.assertEqual(
                    (selected.slot_count, selected.logical_blocks_per_chunk),
                    (slots, 1024),
                )
                self.assertTrue(
                    api.build_memory_plan.call_args.kwargs["warm_cpp_backend"]
                )
                self.assertEqual(
                    calls[0][1],
                    {
                        "backend": expected_backend,
                        "output_slots": banks,
                        "execution": execution,
                        "standard_backend": "cpp",
                    },
                )


if __name__ == "__main__":
    unittest.main()
