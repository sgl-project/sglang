# Copyright 2023-2026 SGLang Team
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
"""The CUDA graph cache configuration surface (design section 13).

Four ``exec.graph`` leaves select graph serialization: ``--cuda-graph-cache-mode``
(default ``off``), ``--cuda-graph-cache-dir``, ``--cuda-graph-cache-placement``
and ``--cuda-graph-cache-verify``. ``apply_cuda_graph_cache_compatibility`` in
``arg_groups/cuda_graph_hook.py`` resolves the mode back to ``off`` with a
logged reason for the configurations v1 does not cover (design sections 9.3
and 9.4).

Pinned here: the default resolves to ``off`` without a declaration; ``save``
survives a supported configuration; every v1 gate declares ``off`` through the
stash (the record keeps the operator's input); the first applicable reason is
the one logged; the CLI parser exposes the four flags with the Literal choices.

    python -m pytest test/registered/unit/server_args/test_cuda_graph_cache_args.py -v
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.cuda_graph_hook import apply_cuda_graph_cache_compatibility
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HOOK_SOURCE = "_apply_cuda_graph_cache_compatibility"
_HOOK_LOGGER = "sglang.srt.arg_groups.cuda_graph_hook"

# A real config.json so the pipeline runs past its dummy-model early return
# (same shape as test_resolution_declarations.py).
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# Every field the hook reads, at values that keep the cache enabled. A test
# flips exactly one of them to reach one gate.
_SUPPORTED = {
    "cuda_graph_cache_mode": "save",
    "cuda_graph_config": None,
    "speculative_algorithm": None,
    "enable_pdmux": False,
    "enable_dp_attention": False,
    "dcp_size": 1,
    "moe_a2a_backend": "none",
    "deepep_mode": "auto",
    "lora_paths": None,
    "enable_lora": None,
    "enable_two_batch_overlap": False,
    "enable_memory_saver": False,
}


def _graph_config(prefill=Backend.BREAKABLE, decode=Backend.FULL):
    return SimpleNamespace(
        prefill=SimpleNamespace(backend=prefill),
        decode=SimpleNamespace(backend=decode),
    )


def _run_hook(**overrides):
    """Run just the hook over a minimal stand-in and return the stand-in."""
    sa = ServerArgs.__new__(ServerArgs)
    fields = dict(_SUPPORTED)
    fields["cuda_graph_config"] = _graph_config()
    fields.update(overrides)
    for name, value in fields.items():
        object.__setattr__(sa, name, value)
    apply_cuda_graph_cache_compatibility(sa)
    return sa


def _declarations(sa):
    return [
        (source, declared)
        for source, declared in getattr(sa, "_resolved_overrides", None) or ()
        if source == _HOOK_SOURCE
    ]


class TestCudaGraphCacheHook(CustomTestCase):
    """The hook in isolation, one gate per test, over a stand-in record."""

    def test_off_returns_before_any_gate(self):
        sa = _run_hook(
            cuda_graph_cache_mode="off",
            enable_two_batch_overlap=True,
            speculative_algorithm="EAGLE",
        )
        self.assertEqual(_declarations(sa), [])

    def test_supported_configuration_keeps_save(self):
        sa = _run_hook()
        self.assertEqual(_declarations(sa), [])
        self.assertEqual(resolution_result(sa, "cuda_graph_cache_mode"), "save")

    def test_none_cuda_graph_config_is_guarded(self):
        sa = _run_hook(cuda_graph_config=None)
        self.assertEqual(_declarations(sa), [])

    def test_each_gate_declares_off_once(self):
        gates = {
            "tc_piecewise prefill": {
                "cuda_graph_config": _graph_config(prefill=Backend.TC_PIECEWISE)
            },
            "tc_piecewise decode": {
                "cuda_graph_config": _graph_config(decode=Backend.TC_PIECEWISE)
            },
            "speculative decoding": {"speculative_algorithm": "EAGLE"},
            "pdmux": {"enable_pdmux": True},
            "dp attention": {"enable_dp_attention": True},
            "dcp": {"dcp_size": 2},
            "deepep low_latency": {
                "moe_a2a_backend": "deepep",
                "deepep_mode": "low_latency",
            },
            "deepep auto": {"moe_a2a_backend": "deepep", "deepep_mode": "auto"},
            "deepep_v2 auto": {"moe_a2a_backend": "deepep_v2"},
            "lora paths": {"lora_paths": ["/tmp/adapter"]},
            "enable_lora": {"enable_lora": True},
            "two-batch overlap": {"enable_two_batch_overlap": True},
            "memory saver": {"enable_memory_saver": True},
        }
        for name, overrides in gates.items():
            for mode in ("save", "load", "auto"):
                with self.subTest(gate=name, mode=mode):
                    sa = _run_hook(cuda_graph_cache_mode=mode, **overrides)
                    self.assertEqual(
                        _declarations(sa),
                        [(_HOOK_SOURCE, {"cuda_graph_cache_mode": "off"})],
                    )
                    self.assertEqual(
                        resolution_result(sa, "cuda_graph_cache_mode"), "off"
                    )
                    # A declaration-only resolver leaves the field alone.
                    self.assertEqual(sa.cuda_graph_cache_mode, mode)

    def test_deepep_normal_mode_is_not_a_gate(self):
        sa = _run_hook(moe_a2a_backend="deepep", deepep_mode="normal")
        self.assertEqual(_declarations(sa), [])

    def test_non_deepep_a2a_backend_ignores_deepep_mode(self):
        sa = _run_hook(moe_a2a_backend="none", deepep_mode="low_latency")
        self.assertEqual(_declarations(sa), [])

    def test_the_first_applicable_reason_is_the_one_logged(self):
        with self.assertLogs(_HOOK_LOGGER, level="WARNING") as captured:
            sa = _run_hook(
                cuda_graph_config=_graph_config(prefill=Backend.TC_PIECEWISE),
                enable_two_batch_overlap=True,
                enable_memory_saver=True,
            )
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertIn("tc_piecewise", message)
        self.assertIn("--cuda-graph-cache-mode save", message)
        self.assertNotIn("two-batch overlap", message)
        # One decision, one declaration.
        self.assertEqual(len(_declarations(sa)), 1)


class TestCudaGraphCacheResolution(CustomTestCase):
    """The hook inside the full pipeline, over a real record."""

    def setUp(self):
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)

    def _resolve(self, **extra):
        path = tempfile.mkdtemp(prefix="cuda_graph_cache_args_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        fields = {"random_seed": 42, "skip_tokenizer_init": True}
        fields.update(extra)
        sa = ServerArgs(model_path=path, device="cuda", **fields)
        sa.resolve_once()
        return sa

    def test_default_resolves_to_off_without_a_declaration(self):
        sa = self._resolve()
        self.assertEqual(resolution_result(sa, "cuda_graph_cache_mode"), "off")
        self.assertEqual(_declarations(sa), [])
        self.assertEqual(resolution_result(sa, "cuda_graph_cache_dir"), None)
        self.assertEqual(
            resolution_result(sa, "cuda_graph_cache_placement"), "relocate"
        )
        self.assertEqual(resolution_result(sa, "cuda_graph_cache_verify"), "none")

    def test_save_survives_breakable_prefill_and_full_decode(self):
        sa = self._resolve(
            cuda_graph_cache_mode="save",
            cuda_graph_backend_prefill="breakable",
            cuda_graph_backend_decode="full",
        )
        self.assertEqual(resolution_result(sa, "cuda_graph_cache_mode"), "save")
        self.assertEqual(_declarations(sa), [])
        graph_config = resolution_result(sa, "cuda_graph_config")
        self.assertEqual(graph_config.prefill.backend, Backend.BREAKABLE)
        self.assertEqual(graph_config.decode.backend, Backend.FULL)

    def test_a_v1_gate_resolves_save_back_to_off(self):
        # Speculative decoding needs no draft model path to resolve on CPU
        # (the same shape test_resolution_declarations.py runs).
        with self.assertLogs(_HOOK_LOGGER, level="WARNING") as captured:
            sa = self._resolve(
                cuda_graph_cache_mode="save",
                cuda_graph_backend_prefill="breakable",
                cuda_graph_backend_decode="full",
                speculative_algorithm="EAGLE",
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
            )
        self.assertEqual(resolution_result(sa, "cuda_graph_cache_mode"), "off")
        self.assertEqual(
            _declarations(sa), [(_HOOK_SOURCE, {"cuda_graph_cache_mode": "off"})]
        )
        # The record keeps the operator's input; the stash carries the decision.
        self.assertEqual(sa.cuda_graph_cache_mode, "save")
        self.assertTrue(
            any(
                "speculative decoding" in record.getMessage()
                for record in captured.records
            ),
            [record.getMessage() for record in captured.records],
        )

    def test_the_published_leaf_carries_the_decision(self):
        from sglang.srt.model_executor.graph_serialization.plan import (
            CacheMode,
            Placement,
            VerifyMode,
            read_plan_from_config,
        )
        from sglang.srt.runtime_context import get_exec, publish, reset_context

        self.addCleanup(reset_context)
        sa = self._resolve(
            cuda_graph_cache_mode="save",
            cuda_graph_backend_prefill="breakable",
            cuda_graph_backend_decode="full",
            cuda_graph_cache_dir="/nonexistent/cuda_graphs",
            cuda_graph_cache_placement="fixed_va",
            cuda_graph_cache_verify="shadow-one",
        )
        publish(sa, role="scheduler")
        graph = get_exec().graph
        self.assertEqual(graph.cuda_graph_cache_mode, "save")
        self.assertEqual(graph.cuda_graph_cache_dir, "/nonexistent/cuda_graphs")
        self.assertEqual(graph.cuda_graph_cache_placement, "fixed_va")
        self.assertEqual(graph.cuda_graph_cache_verify, "shadow-one")
        plan = read_plan_from_config()
        self.assertIs(plan.mode, CacheMode.SAVE)
        self.assertEqual(plan.cache_dir, "/nonexistent/cuda_graphs")
        self.assertIs(plan.placement, Placement.FIXED_VA)
        self.assertIs(plan.verify, VerifyMode.SHADOW_ONE)


class TestCudaGraphCacheCli(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(self.parser)

    def _action(self, dest):
        return next(action for action in self.parser._actions if action.dest == dest)

    def test_mode_flag_exposes_the_literal_choices(self):
        action = self._action("cuda_graph_cache_mode")
        self.assertIn("--cuda-graph-cache-mode", action.option_strings)
        self.assertEqual(list(action.choices), ["off", "save", "load", "auto"])
        self.assertEqual(action.default, "off")

    def test_dir_flag_is_an_optional_string(self):
        action = self._action("cuda_graph_cache_dir")
        self.assertIn("--cuda-graph-cache-dir", action.option_strings)
        self.assertIsNone(action.default)
        self.assertIsNone(action.choices)

    def test_placement_flag_exposes_the_literal_choices(self):
        action = self._action("cuda_graph_cache_placement")
        self.assertIn("--cuda-graph-cache-placement", action.option_strings)
        self.assertEqual(list(action.choices), ["relocate", "fixed_va"])
        self.assertEqual(action.default, "relocate")

    def test_verify_flag_exposes_the_literal_choices(self):
        action = self._action("cuda_graph_cache_verify")
        self.assertIn("--cuda-graph-cache-verify", action.option_strings)
        self.assertEqual(list(action.choices), ["none", "shadow-one", "shadow-all"])
        self.assertEqual(action.default, "none")

    def test_the_flags_parse_into_the_record(self):
        args = self.parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--cuda-graph-cache-mode",
                "load",
                "--cuda-graph-cache-dir",
                "/tmp/graphs",
                "--cuda-graph-cache-placement",
                "fixed_va",
                "--cuda-graph-cache-verify",
                "shadow-all",
            ]
        )
        sa = ServerArgs.from_cli_args(args)
        self.assertEqual(sa.cuda_graph_cache_mode, "load")
        self.assertEqual(sa.cuda_graph_cache_dir, "/tmp/graphs")
        self.assertEqual(sa.cuda_graph_cache_placement, "fixed_va")
        self.assertEqual(sa.cuda_graph_cache_verify, "shadow-all")

    def test_an_unknown_mode_is_rejected_by_the_parser(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["--model-path", "dummy", "--cuda-graph-cache-mode", "maybe"]
            )


if __name__ == "__main__":
    unittest.main()
