"""--engram-config: parsing, the deprecated environment variables it replaces,
and the resolved value the Engram layers read."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import _resolve_engram_config
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.layers.engram_config import EngramConfig
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GPU = {"placement": "gpu", "replication": "none", "lookup": "collective"}
CPU_SHARDED = {"placement": "cpu", "replication": "none", "lookup": "collective"}
CPU_PER_NODE = {"placement": "cpu", "replication": "per_node", "lookup": "collective"}
PEER_READ = {"placement": "gpu", "replication": "none", "lookup": "peer_read"}


def _args(model_type="deepseek_v41", **overrides):
    args = ServerArgs(**{"model_path": "local-deepseek-v4", "tp_size": 2, **overrides})
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], model_type=model_type
        ),
    )
    return args


def _resolve(env=None, **overrides):
    args = _args(**overrides)
    with ExitStack() as stack:
        for name, value in (env or {}).items():
            stack.enter_context(getattr(envs, name).override(value))
        _resolve_engram_config(args)
    return resolution_result(args, "engram_config")


class TestEngramConfig(CustomTestCase):
    def setUp(self):
        self.enterContext(override_platform(is_cuda=True, is_hip=False))

    def test_partial_json_keeps_defaults(self):
        cases = (
            (None, GPU),
            ({"placement": "cpu"}, CPU_SHARDED),
            ({"placement": "cpu", "replication": "per_node"}, CPU_PER_NODE),
            ({"lookup": "peer_read"}, PEER_READ),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_resolve(engram_config=raw), expected)

    def test_invalid_config_rejected(self):
        cases = (
            ("unknown key", {"layout": "shared"}),
            ("expected one of", {"placement": "host"}),
            ("needs placement 'cpu'", {"replication": "per_node"}),
            (
                "lookup applies only",
                {"placement": "cpu", "replication": "per_node", "lookup": "collective"},
            ),
            ("needs placement 'gpu'", {"placement": "cpu", "lookup": "peer_read"}),
            ("JSON object", ["cpu"]),
        )
        for regex, raw in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, regex):
                EngramConfig.from_dict(raw)

    def test_topology_rejections(self):
        with self.assertRaisesRegex(ValueError, "one node"):
            _resolve(
                engram_config={"placement": "cpu", "replication": "per_node"},
                nnodes=2,
            )
        with (
            override_platform(is_cuda=False, is_hip=True),
            self.assertRaisesRegex(ValueError, "needs CUDA"),
        ):
            _resolve(engram_config={"lookup": "peer_read"})
        with self.assertRaisesRegex(ValueError, "DeepSeek-V4.1 only"):
            _resolve(model_type="deepseek_v4", engram_config={"placement": "cpu"})

    def test_deprecated_environment_variables(self):
        host = {"SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": True}
        cases = (
            # The layout used to default to shared; unset it is now per_rank.
            (host, CPU_SHARDED),
            ({**host, "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared"}, CPU_PER_NODE),
            (
                {**host, "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank"},
                CPU_SHARDED,
            ),
            # Peer shards take precedence over the host table, as before.
            ({**host, "SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS": True}, PEER_READ),
            ({"SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared"}, GPU),
        )
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(_resolve(env), expected)
        # Without TP there is no peer to read; the host table applies, as before.
        self.assertEqual(
            _resolve({**host, "SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS": True}, tp_size=1),
            CPU_SHARDED,
        )
        with (
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("numa"),
            self.assertRaisesRegex(ValueError, "Invalid SGLANG_DSV41"),
        ):
            _resolve_engram_config(_args())

    def test_explicit_config_wins_over_environment(self):
        # Ignored variables are not translated, so an invalid one does not raise.
        env = {
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": True,
            "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "numa",
        }
        self.assertEqual(_resolve(env, engram_config={}), GPU)


if __name__ == "__main__":
    unittest.main()
