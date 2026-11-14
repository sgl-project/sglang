# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the plugin-facing subset of ``RuntimeContext``.

Out-of-tree plugins validate active runtime settings inside scheduler /
model-runner hooks. Because ``ServerArgs`` retains raw caller input after
resolution, plugins rely on the documented ``sglang.srt.runtime_context``
accessors and fields remaining importable, published for the ``"scheduler"``
role, and reflective of resolution decisions (such as per-phase CUDA graph
backends and ``--lora-paths`` implying ``enable_lora``).

Run: python -m pytest test/registered/unit/plugins/test_plugin_runtime_context_contract.py -v
"""

import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    ROLE_NAMESPACE_SETS,
    SpawnRanks,
    get_device,
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_mm,
    get_model,
    get_observability,
    get_parallel,
    get_schedule,
    get_server_args,
    get_serving,
    get_spec,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 128,
    "intermediate_size": 256,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
    "vocab_size": 1000,
}

_DOCUMENTED_NAMESPACES = {
    "device": get_device,
    "disagg": get_disagg,
    "exec": get_exec,
    "lora": get_lora,
    "memory": get_memory,
    "mm": get_mm,
    "model": get_model,
    "observability": get_observability,
    "schedule": get_schedule,
    "serving": get_serving,
    "spec": get_spec,
}


class TestPluginRuntimeContextContract(CustomTestCase):
    def setUp(self):
        super().setUp()
        saved_env = dict(os.environ)

        def restore_env():
            os.environ.clear()
            os.environ.update(saved_env)

        self.addCleanup(restore_env)
        self.addCleanup(reset_context)
        reset_context()

    def _model_path(self) -> str:
        path = tempfile.mkdtemp(prefix="plugin_ctx_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(_MINI_CONFIG, f)
        return path

    def _resolve_and_publish(
        self,
        *,
        role: str = "scheduler",
        ranks: SpawnRanks | None = None,
        **kwargs,
    ) -> ServerArgs:
        server_args = ServerArgs(
            model_path=self._model_path(),
            device="cuda",
            random_seed=42,
            **kwargs,
        )
        server_args.resolve_once()
        server_args.check_server_args()
        publish(server_args, role=role, ranks=ranks)
        return server_args

    def test_accessors_fail_closed_before_publish(self):
        """load_plugins() runs before publish(); bag reads must fail closed."""
        for name, getter in _DOCUMENTED_NAMESPACES.items():
            with self.subTest(namespace=name):
                with self.assertRaises(ValueError):
                    getter()

    def test_scheduler_role_permits_all_documented_namespaces(self):
        """Model-runner and scheduler hooks run under role='scheduler'."""
        self.assertIn("scheduler", ROLE_NAMESPACE_SETS)
        allowed = ROLE_NAMESPACE_SETS["scheduler"]
        if allowed is not None:
            self.assertTrue(
                set(_DOCUMENTED_NAMESPACES).issubset(allowed),
                f"scheduler role missing namespaces: {set(_DOCUMENTED_NAMESPACES) - allowed}",
            )

        self._resolve_and_publish()
        for name, getter in _DOCUMENTED_NAMESPACES.items():
            with self.subTest(namespace=name):
                self.assertIsNotNone(getter())

    def test_cuda_graph_backend_resolves_across_cli_and_dict_inputs(self):
        """Raw ServerArgs keeps None or dict; get_exec() exposes resolved phase configs."""
        # Default launch: raw cuda_graph_config is None, resolved decode backend is set.
        self._resolve_and_publish()
        self.assertIsNone(get_server_args().cuda_graph_config)
        graph_cfg = get_exec().graph.cuda_graph_config
        self.assertIsNotNone(graph_cfg)
        self.assertEqual(graph_cfg.decode.backend, Backend.FULL)
        self.assertFalse(get_exec().graph.enable_torch_compile)

        reset_context()
        # Engine-style dict input: raw stays dict, bag holds structured config.
        self._resolve_and_publish(
            cuda_graph_config={"decode": {"backend": "disabled", "max_bs": 8}},
            enable_torch_compile=True,
        )
        self.assertIsInstance(get_server_args().cuda_graph_config, dict)
        self.assertEqual(
            get_exec().graph.cuda_graph_config.decode.backend, Backend.DISABLED
        )
        self.assertEqual(get_exec().graph.cuda_graph_config.decode.max_bs, 8)
        self.assertTrue(get_exec().graph.enable_torch_compile)

    def test_lora_paths_imply_enable_lora_in_context_not_raw_args(self):
        """--lora-paths without --enable-lora resolves enable_lora=True only in get_lora()."""
        self._resolve_and_publish(lora_paths=["adapter=/tmp/nonexistent-lora"])
        self.assertIsNone(get_server_args().enable_lora)
        self.assertTrue(get_lora().enable_lora)

    def test_parallel_widths_and_spawn_ranks(self):
        """Documented parallel widths and ranks resolve consistently on get_parallel()."""
        self._resolve_and_publish(
            tp_size=4,
            dcp_size=2,
            ranks=SpawnRanks(world_rank=3, dp_rank=1, gpu_id=3),
        )
        p = get_parallel()
        self.assertEqual(p.tp_size, 4)
        self.assertEqual(p.pp_size, 1)
        self.assertEqual(p.dp_size, 1)
        self.assertEqual(p.dcp_size, 2)
        self.assertTrue(p.dcp_enabled)
        self.assertEqual(p.attn_tp_size, 4)
        self.assertEqual(p.attn_dp_size, 1)
        self.assertEqual(p.moe_ep_size, 1)
        self.assertEqual(p.moe_dp_size, 1)
        self.assertEqual(p.tp_rank, 3)
        self.assertEqual(p.pp_rank, 0)
        self.assertEqual(p.dp_rank, 1)
        self.assertEqual(p.dcp_rank, 1)


if __name__ == "__main__":
    unittest.main()
