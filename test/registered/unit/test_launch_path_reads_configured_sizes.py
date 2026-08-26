"""A parallel size has one spelling.

`get_parallel().tp_size` used to answer what the process ended up in while
`get_parallel().config.tp_size` answered what the launch asked for, and the
launch paths had to remember which one they wanted -- a size read before
distributed init raised, and a size read after it could disagree with the
configuration. The two answers were the same number: the group is created with
exactly the configured width, in all four `initialize_model_parallel` call
sites, and `ensure_model_parallel_initialized` asserts it.

So the sizes read bare and answer from the published configuration, which is
available before the groups exist. The one member that genuinely disagrees
carries its own name: `moe_dp_group_size` is the width of the group the MoE
all-gather runs over, which `initialize_model_parallel` aliases to the
attention-CP group when `attn_cp_size > moe_dp_size`.

This pins both halves: no size may be spelled two ways again, and the one that
is deliberately two members keeps two names.
"""

import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

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


def _shadowed() -> set:
    """Names that are both a live ``ParallelContext`` member and a config leaf.

    Derived from the two sides, so a member added on either side is watched
    without a second list here.
    """
    from sglang.srt.arg_groups.arg_utils import namespace_of
    from sglang.srt.runtime_context import ParallelContext
    from sglang.srt.server_args import ServerArgs

    live = {
        name
        for name, value in vars(ParallelContext).items()
        if isinstance(value, property)
    }
    leaves = {
        field for field, path in namespace_of(ServerArgs).items() if path == "parallel"
    }
    return live & leaves


class TestOneSpellingPerParallelSize(CustomTestCase):
    def setUp(self):
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)

    def _published(self, **supplied):
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        directory = tempfile.mkdtemp(prefix="one_spelling_")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        self.addCleanup(reset_context)
        with open(os.path.join(directory, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        server_args = ServerArgs(model_path=directory, device="cuda", **supplied)
        server_args.resolve_once()
        publish(server_args, role="scheduler")
        return server_args

    def test_no_size_is_spelled_two_ways(self):
        shadowed = _shadowed()
        self.assertEqual(
            set(),
            shadowed,
            "these names are both a live member and a config leaf, so a reader "
            "cannot tell which one it got; give the live member its own name "
            f"the way moe_dp_group_size has one: {sorted(shadowed)}",
        )

    def test_a_size_reads_bare_and_answers_from_the_configuration(self):
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.runtime_context import get_parallel

        server_args = self._published(tp_size=2)
        for name in ("tp_size", "pp_size", "attn_cp_size", "dcp_size", "dp_size"):
            with self.subTest(size=name):
                self.assertEqual(
                    resolution_result(server_args, name),
                    getattr(get_parallel(), name),
                    f"the bare {name} disagreed with what resolution decided",
                )

    def test_the_aliased_group_width_keeps_its_own_name(self):
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.runtime_context import get_parallel

        server_args = self._published()
        configured = resolution_result(server_args, "moe_dp_size")
        with patch(
            "sglang.srt.distributed.parallel_state.get_moe_data_parallel_world_size",
            return_value=configured + 41,
        ):
            self.assertEqual(
                configured + 41,
                get_parallel().moe_dp_group_size,
                "moe_dp_group_size stopped following the live group, which is "
                "the only reason it is a separate member",
            )
            self.assertEqual(
                configured,
                get_parallel().moe_dp_size,
                "the bare moe_dp_size followed the live group instead of the "
                "configuration",
            )

    def test_an_unpublished_read_says_so(self):
        from sglang.srt.runtime_context import ParallelContext

        with self.assertRaisesRegex(ValueError, r"'parallel' not published"):
            ParallelContext().tp_size
        with self.assertRaisesRegex(AttributeError, r"has no 'not_a_leaf'"):
            ParallelContext().not_a_leaf


if __name__ == "__main__":
    unittest.main()
