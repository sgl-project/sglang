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
"""Every spec worker takes the call the scheduler makes.

BUG REGRESSION. `Scheduler.run_batch` drives whichever V2 spec worker
`SpeculativeAlgorithm.create_worker` installed, and on the non-overlap branch it
passes `pp_proxy_tensors=` unconditionally. Three workers never grew the
parameter -- DSpark, frozen-KV MTP and multi-layer EAGLE -- so any run on a
non-overlap schedule died with

    TypeError: DSparkWorkerV2.forward_batch_generation() got an unexpected
               keyword argument 'pp_proxy_tensors'

on BOTH the baseline and the unified arm (eval_571 lost the whole Kimi x DSPARK
cell that way; Kimi pins normal scheduling, so it takes that branch every time).

The worker set is DERIVED from `create_worker`'s own `return` statements, so a
worker added later is covered without editing this file -- a hand-kept list
would pass forever while the new worker drifted.

    python -m pytest test/registered/unit/spec/test_spec_worker_forward_signature.py -v
"""

import ast
import pathlib
import unittest

import sglang.srt.speculative.spec_info as spec_info
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# What Scheduler.run_batch passes on the non-overlap spec branch, beyond the
# positional batch. Keep in sync with scheduler.py's call, not with any worker.
_SCHEDULER_KWARGS = ("pp_proxy_tensors",)


def _worker_classes():
    """(class_name, module_path) for every worker `create_worker` can return.

    AST rather than import: several worker modules pull in compiled kernels that
    are absent on a CPU box, and a skip here would be a silent coverage hole in
    exactly the guard that is supposed to be un-skippable.
    """
    tree = ast.parse(pathlib.Path(spec_info.__file__).read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "create_worker"
    )
    modules = {
        alias.name: node.module
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    returned = [
        node.value.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    ]
    return [(name, modules[name]) for name in returned if name in modules]


def _forward_params(module_path: str, class_name: str):
    root = pathlib.Path(spec_info.__file__).parents[3]
    src = root / (module_path.replace(".", "/") + ".py")
    tree = ast.parse(src.read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    fn = next(
        (
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_batch_generation"
        ),
        None,
    )
    if fn is None:
        return None
    args = fn.args
    names = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
    if args.kwarg is not None:
        names.add("**")
    return names


class TestSpecWorkerForwardSignature(CustomTestCase):
    def test_every_worker_accepts_the_scheduler_call(self):
        workers = _worker_classes()
        self.assertGreaterEqual(
            len(workers), 5, "create_worker's return set failed to resolve"
        )
        missing = []
        for class_name, module_path in workers:
            params = _forward_params(module_path, class_name)
            if params is None:
                # Inherits forward_batch_generation from a base; the base is
                # itself a worker the scheduler drives, so nothing to pin here.
                continue
            if "**" in params:
                continue
            for kwarg in _SCHEDULER_KWARGS:
                if kwarg not in params:
                    missing.append(f"{class_name}.forward_batch_generation({kwarg}=)")
        self.assertEqual(
            missing,
            [],
            "these spec workers cannot bind the call Scheduler.run_batch makes, "
            "so every non-overlap spec run on them dies with a TypeError: "
            + ", ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
