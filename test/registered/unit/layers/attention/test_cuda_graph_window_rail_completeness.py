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
"""The multi-step draft's sliding-window rails.

Every Triton cuda-graph forward mode must fill the sliding-window READ rail,
and each draft step of every multi-step draft backend must write its window
layers through its OWN columns of the iteration's write window.

BUG REGRESSION. `_build_cuda_graph_forward_metadata` builds one
`ForwardMetadata` per captured forward mode. The draft-extend branch passed
`window_kv_indices=None` while its buffer-fill helper never built the rail,
so a banded MTP depth (a window layer on the draft) reached the Triton extend
kernel with a null index tensor and died during graph capture as
`AttributeError("'NoneType' object has no attribute 'type'")` inside IR
construction -- a message that names neither the rail nor the mode. The eager
path built it, so only cuda-graph runs failed.

A bare `None` here is the defect: every mode this function serves can carry
window layers, so each must pass the buffer, guarded on the model actually
having a window. This is an AST check over the real function body, so a new
mode branch that forgets the rail fails here rather than on a GPU.

    python -m pytest test/registered/unit/layers/attention/test_cuda_graph_window_rail_completeness.py -v
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.iteration_ids import IterationIds
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_BUILDER = "_build_cuda_graph_forward_metadata"
_RAIL = "window_kv_indices"


def _builder_body():
    import sglang.srt.layers.attention.triton_backend as tb

    tree = ast.parse(Path(tb.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _BUILDER:
            return node
    raise AssertionError(f"{_BUILDER} not found in triton_backend.py")


def _rail_args(body):
    """(line, value node) of the window-rail kwarg of each ForwardMetadata."""
    for call in ast.walk(body):
        if not isinstance(call, ast.Call):
            continue
        name = call.func.id if isinstance(call.func, ast.Name) else None
        if name != "ForwardMetadata":
            continue
        for kw in call.keywords:
            if kw.arg == _RAIL:
                yield call.lineno, kw.value
                break
        else:
            yield call.lineno, None  # kwarg omitted entirely


class TestCudaGraphWindowRailCompleteness(CustomTestCase):
    def test_no_captured_mode_passes_a_bare_none_rail(self):
        rails = list(_rail_args(_builder_body()))
        # A scan that finds nothing would pass vacuously forever.
        self.assertGreaterEqual(
            len(rails), 3, "expected one ForwardMetadata per captured forward mode"
        )
        offenders = [
            line
            for line, value in rails
            if value is None
            or (isinstance(value, ast.Constant) and value.value is None)
        ]
        self.assertEqual(
            offenders,
            [],
            f"{_BUILDER}: {_RAIL} is a bare None at line(s) {offenders}; a window "
            "layer in that mode reaches the Triton kernel with a null index "
            "tensor. Pass the cuda-graph buffer, guarded on the sliding-window flag.",
        )


class _Reader:
    """A reader whose sliding-window ids are the virtual ids times ten."""

    def __init__(self):
        self._spaces = {
            IdSpaceKind.FULL: IdSpace(key="full"),
            IdSpaceKind.SLIDING_WINDOW: IdSpace(key="swa", write=lambda ids: ids * 10),
        }

    def space(self, kind):
        return self._spaces.get(kind)


class TestDraftStepWriteColumns(CustomTestCase):
    """BUG REGRESSION. A draft step's per-step backend took its swa write ids
    from the whole draft window, and the write then kept their first
    ``bs * topk``: step 0's slots for request 0, step 1's for request 0, and so
    on, since the window lays each row out step-minor. Each step must bind its
    own columns while its backend derives the ids."""

    def _backends(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferMultiStepDraftBackend,
        )
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return (
            TritonMultiStepDraftBackend,
            FlashInferMultiStepDraftBackend,
            TRTLLMHAAttnMultiStepDraftBackend,
        )

    def _batch(self, *, bs, topk, num_steps, window_cols):
        reader = _Reader()
        plan = IterationIds(
            source=reader,
            req_pool_indices=torch.arange(bs),
            seq_lens=torch.ones(bs, dtype=torch.int64),
            seq_lens_cpu=None,
            write_virtual=torch.arange(1, bs * topk * window_cols + 1),
            read_extent=0,
        )
        # A chain draft (topk 1) inside a wider verify window writes its
        # leading columns; otherwise the draft owns the whole window.
        cols = None if window_cols == num_steps else slice(0, num_steps)
        batch = SimpleNamespace(
            batch_size=bs,
            iteration_ids=plan,
            iteration_cols=cols,
            out_cache_loc=plan.write_ids(reader, cols=cols),
        )
        swa = plan.write_ids(reader, kind=IdSpaceKind.SLIDING_WINDOW, cols=cols)
        per_step = per_step_draft_out_cache_loc(swa, bs, topk, num_steps)
        # The whole window's leading ids are not step 1's slots.
        self.assertFalse(torch.equal(swa[: bs * topk], per_step[1]))
        return reader, plan, batch, cols, per_step

    def _backend(self, cls, step_backends, *, topk, num_steps):
        backend = object.__new__(cls)
        backend.topk = topk
        backend.speculative_num_steps = num_steps
        backend.attn_backends = step_backends
        return backend

    def _check(self, *, bs, topk, num_steps, window_cols):
        reader, plan, batch, cols, per_step = self._batch(
            bs=bs, topk=topk, num_steps=num_steps, window_cols=window_cols
        )
        for cls in self._backends():
            backend = self._backend(
                cls,
                [SimpleNamespace(use_sliding_window_kv_pool=True)] * num_steps,
                topk=topk,
                num_steps=num_steps,
            )
            for step in range(num_steps):
                with self.subTest(backend=cls.__name__, step=step):
                    with backend._step_writes(batch, step):
                        ids = plan.write_ids(
                            reader,
                            kind=IdSpaceKind.SLIDING_WINDOW,
                            cols=batch.iteration_cols,
                        )
                        torch.testing.assert_close(ids, per_step[step], rtol=0, atol=0)
                        torch.testing.assert_close(
                            batch.out_cache_loc * 10, ids, rtol=0, atol=0
                        )
                    self.assertIs(batch.iteration_cols, cols)

    def test_trtllm_steps_init_from_their_own_columns(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        bs, topk, num_steps = 3, 1, 4
        reader, plan, batch, cols, per_step = self._batch(
            bs=bs, topk=topk, num_steps=num_steps, window_cols=num_steps
        )
        seen = []

        def step_backend():
            def init(fb):
                seen.append(
                    plan.write_ids(
                        reader, kind=IdSpaceKind.SLIDING_WINDOW, cols=fb.iteration_cols
                    )
                )

            return SimpleNamespace(
                use_sliding_window_kv_pool=True, init_forward_metadata=init
            )

        backend = self._backend(
            TRTLLMHAAttnMultiStepDraftBackend,
            [step_backend() for _ in range(num_steps)],
            topk=topk,
            num_steps=num_steps,
        )
        backend.init_forward_metadata(batch)
        self.assertEqual(len(seen), num_steps - 1)
        for step, ids in enumerate(seen):
            torch.testing.assert_close(ids, per_step[step], rtol=0, atol=0)
        self.assertIs(batch.iteration_cols, cols)

    def test_trtllm_steps_capture_their_own_contiguous_writes(self):
        """BUG REGRESSION. Under CUDA graphs a static window pool's per-step
        backend derives its window write ids in a captured metadata kernel
        that reads ``bs * topk`` ids by contiguous offsets from the batch's
        write ids. Every step was handed the whole draft window, so each
        wrote its window layers at the first step's slots (request 0's steps
        for two requests). Each step must capture its own ids, contiguous."""
        from unittest.mock import patch

        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )
        from sglang.srt.model_executor import forward_batch_info

        for bs, topk, num_steps in ((2, 1, 3), (3, 1, 4), (2, 2, 3)):
            with self.subTest(bs=bs, topk=topk, num_steps=num_steps):
                batch = SimpleNamespace(
                    batch_size=bs,
                    iteration_ids=None,
                    iteration_cols=None,
                    out_cache_loc=torch.arange(1, bs * topk * num_steps + 1),
                    spec_info=SimpleNamespace(is_draft_input=lambda: True),
                    encoder_lens=None,
                )
                expected = per_step_draft_out_cache_loc(
                    batch.out_cache_loc, bs, topk, num_steps
                )
                read = []

                def step_backend():
                    def init_in_graph(fb):
                        loc = fb.out_cache_loc
                        # What the kernel reads: n ids from the data pointer.
                        read.append(loc.as_strided((bs * topk,), (1,)).clone())

                    return SimpleNamespace(
                        use_sliding_window_kv_pool=True,
                        init_forward_metadata_in_graph=init_in_graph,
                    )

                backend = self._backend(
                    TRTLLMHAAttnMultiStepDraftBackend,
                    [step_backend() for _ in range(num_steps)],
                    topk=topk,
                    num_steps=num_steps,
                )
                with patch.object(
                    forward_batch_info, "build_inner_fb_view", lambda fb, **kw: fb
                ):
                    backend.init_forward_metadata_in_graph(batch)
                self.assertEqual(len(read), num_steps - 1)
                for step, ids in enumerate(read):
                    torch.testing.assert_close(ids, expected[step], rtol=0, atol=0)

    def test_the_draft_window(self):
        self._check(bs=3, topk=1, num_steps=3, window_cols=3)
        self._check(bs=2, topk=2, num_steps=3, window_cols=3)

    def test_a_chain_draft_inside_the_verify_window(self):
        self._check(bs=3, topk=1, num_steps=3, window_cols=5)


if __name__ == "__main__":
    unittest.main()
