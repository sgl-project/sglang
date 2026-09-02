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
"""`resolve_spec_aux_hidden_state_config` must not read parallel state.

BUG REGRESSION. `ModelRunner.init_spec_aux_hidden_state()` runs BEFORE
`init_torch_distributed()`, so the attention-TP group does not exist while
this resolver runs. A `get_parallel().attn_tp_size` read inside it therefore
raises `AssertionError: attention tensor model parallel group is not
initialized` and kills the server at startup — for EVERY EAGLE-family run
with a draft path, unified pool or not. It is invisible to CPU unit gates
(they never boot a runner) and was caught only by a GPU boot guard.

The draft's kv-head count is therefore stored UNDIVIDED, and consumers apply
attn_tp at pool-build time (`KVCacheConfigurator.fused_draft_kv_region`),
where the group exists — the same point the target divides its own heads.

    python -m pytest test/registered/unit/model_executor/test_spec_aux_hidden_state_boot_order.py -v
"""

import ast
import unittest
from pathlib import Path

import sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state as _mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRC = Path(_mod.__file__)


def _function_named(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_SRC}")


class TestResolverTouchesNoParallelState(unittest.TestCase):
    def test_no_get_parallel_in_the_eagle_resolver(self):
        """AST pin: the EAGLE resolver runs pre-distributed-init, so any
        parallel-state read in it is a boot crash, not a wrong number."""
        tree = ast.parse(_SRC.read_text())
        fn = _function_named(tree, "_resolve_eagle_aux_hidden_state")
        called = {
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertNotIn(
            "get_parallel",
            called,
            "_resolve_eagle_aux_hidden_state reads parallel state, but it runs "
            "before init_torch_distributed() — the attention-TP group does not "
            "exist yet and the server dies at startup.",
        )
        imported = {
            alias.name
            for n in ast.walk(fn)
            if isinstance(n, ast.ImportFrom)
            for alias in n.names
        }
        self.assertNotIn("get_parallel", imported)

    def test_nextn_layers_come_from_a_draft_model_config(self):
        """BUG REGRESSION (eval_568). Every `num_nextn_predict_layers = 1`
        assignment in model_config.py is guarded by `if is_draft_model`, so the
        TARGET's own config always answers None. Reading it off the target made
        the path-less-NEXTN geometry silently never resolve, and the draft fell
        back to a private pool instead of fusing. The resolver must therefore
        build a config with is_draft_model=True and read the field off THAT --
        never off the target `model_config` parameter."""
        tree = ast.parse(_SRC.read_text())
        fn = _function_named(tree, "_resolve_eagle_aux_hidden_state")
        target_reads = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Attribute)
            and n.attr == "num_nextn_predict_layers"
            and isinstance(n.value, ast.Name)
            and n.value.id == "model_config"
        ]
        self.assertEqual(
            target_reads,
            [],
            "_resolve_eagle_aux_hidden_state reads num_nextn_predict_layers off "
            "the TARGET model_config, where it is always None (the field is "
            "only filled under is_draft_model=True).",
        )
        self.assertIn(
            "is_draft_model=True",
            ast.unparse(fn),
            "the resolver must build the draft config with is_draft_model=True",
        )

    def test_the_recorded_head_count_is_undivided(self):
        """The field name must keep saying TOTAL: a consumer that reads it as
        a per-GPU count silently under-sizes the fused draft region at
        attn_tp > 1 (and every unit fixture runs at tp=1, where the bug is
        invisible)."""
        fields = _mod.SpecAuxHiddenStateConfig.__struct_fields__
        self.assertIn("eagle_draft_total_kv_heads", fields)
        self.assertNotIn("eagle_draft_kv_head_num", fields)


if __name__ == "__main__":
    unittest.main()
