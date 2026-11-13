"""DWDP resolves with every CUDA graph phase off, because its prefetch remaps pages on the host between layers."""

import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


class TestDwdpDisablesGraphs(CustomTestCase):
    def _resolve(self, **fields) -> ServerArgs:
        environment = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(environment)))
        path = tempfile.mkdtemp(prefix="dwdp_args_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        server_args = ServerArgs(
            model_path=path, device="cuda", tp_size=2, dwdp_size=2, **fields
        )
        server_args.resolve_once()
        return server_args

    def test_both_graph_phases_resolve_disabled(self) -> None:
        graphs = resolution_result(self._resolve(), "cuda_graph_config")
        self.assertEqual(graphs.decode.backend, "disabled")
        self.assertEqual(graphs.prefill.backend, "disabled")

    def test_a_requested_graph_phase_is_rejected(self) -> None:
        for field in ("cuda_graph_backend_decode", "cuda_graph_backend_prefill"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(AssertionError, "without CUDA graphs"):
                    self._resolve(**{field: "breakable"})


if __name__ == "__main__":
    unittest.main()
