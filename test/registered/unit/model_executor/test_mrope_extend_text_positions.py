"""An all-text extend's MRoPE rows built on the device.

ForwardBatch._compute_mrope_positions_extend builds each text request's rows as
arange(prefix, prefix + extend) on the host and copies them to the device from
pageable memory. For an all-text batch the rows equal the 1-D positions
compute_position already wrote on the device from the same host lists, so they
are repeated there instead; every other batch keeps the host loop.
"""

import random
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.model_executor.forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HAS_CUDA = torch.cuda.is_available()


def _positions(prefix, extend, device="cpu"):
    return fbi.compute_position_torch(
        torch.tensor(prefix, dtype=torch.int32, device=device),
        torch.tensor(extend, dtype=torch.int32, device=device),
    )[0]


def _extend(prefix, extend, *, positions=None, mm=None, host_lists=True, dllm=None):
    """A stand-in ForwardBatch and ScheduleBatch for one extend."""
    fb = object.__new__(ForwardBatch)
    fb.positions = _positions(prefix, extend) if positions is None else positions
    fb.extend_seq_lens_cpu = list(extend) if host_lists else None
    fb.extend_prefix_lens_cpu = list(prefix) if host_lists else None
    fb.seq_lens_cpu = torch.tensor([p + e for p, e in zip(prefix, extend)])
    fb.mrope_positions = None
    batch = SimpleNamespace(
        extend_lens=list(extend),
        prefix_lens=list(prefix),
        multimodal_inputs=mm if mm is not None else [None] * len(extend),
        dllm_config=dllm,
        reqs=[SimpleNamespace(session=None) for _ in extend],
    )
    return fb, batch


def _rows(prefix, extend, *, device="cpu", rl=None, host_loop=False, **kwargs):
    fb, batch = _extend(prefix, extend, **kwargs)
    exec_config = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=rl))
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(fbi, "get_exec", return_value=exec_config)
        )
        if host_loop:
            stack.enter_context(
                mock.patch.object(
                    ForwardBatch, "_text_only_mrope_positions", return_value=None
                )
            )
        fb._compute_mrope_positions_extend(SimpleNamespace(device=device), batch)
    return fb.mrope_positions


def _mm(n):
    return SimpleNamespace(mrope_positions=torch.randint(0, 1 << 20, (3, n)))


class TestMropeExtendTextPositions(CustomTestCase):
    def assert_same(self, got, want):
        self.assertEqual(
            (got.dtype, got.shape, got.device), (want.dtype, want.shape, want.device)
        )
        self.assertTrue(got.is_contiguous())
        self.assertTrue(torch.equal(got, want))

    def test_text_rows_equal_the_host_loop(self):
        rng = random.Random(0)
        layouts = [([0, 0, 0], [5, 3, 7]), ([5, 9, 0], [0, 4, 3]), ([0], [32768])]
        for _ in range(100):
            bs = rng.randint(1, 64)
            prefix = [rng.choice([0, rng.randint(1, 262144)]) for _ in range(bs)]
            extend = [rng.choice([0, 1, rng.randint(1, 64)]) for _ in range(bs)]
            extend[0] = max(extend[0], 1)
            layouts.append((prefix, extend))
        for prefix, extend in layouts:
            with self.subTest(bs=len(prefix)):
                fb, batch = _extend(prefix, extend)
                self.assertIsNotNone(
                    fb._text_only_mrope_positions(batch=batch, rl_on_policy_target=None)
                )
                self.assert_same(
                    _rows(prefix, extend), _rows(prefix, extend, host_loop=True)
                )

    def test_rows_are_read_from_the_batch_positions(self):
        # Exact only because compute_position built them from the same lists.
        shifted = _positions([3, 0], [4, 2]) + 1
        rows = _rows([3, 0], [4, 2], positions=shifted)
        self.assertTrue(torch.equal(rows, shifted.unsqueeze(0).repeat(3, 1)))

    def test_rl_on_policy_target_treats_every_row_as_text(self):
        prefix, extend = [0, 7, 0], [5, 40, 6]
        mm = [None, _mm(47), None]
        fb, batch = _extend(prefix, extend, mm=mm)
        self.assertIsNotNone(
            fb._text_only_mrope_positions(batch=batch, rl_on_policy_target="fsdp")
        )
        self.assert_same(
            _rows(prefix, extend, mm=mm, rl="fsdp"),
            _rows(prefix, extend, mm=mm, rl="fsdp", host_loop=True),
        )

    def test_other_batches_keep_the_host_loop(self):
        padded = torch.cat([torch.arange(8), torch.full((8,), -7)])
        cases = {
            "multimodal row": dict(mm=[None, _mm(40), None]),
            "padded positions": dict(positions=padded),
            "device-only lens": dict(host_lists=False),
            "dllm": dict(dllm=object()),
            "2-D positions": dict(positions=_positions([0, 0, 0], [5, 40, 6])[None]),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                prefix, extend = [0, 0, 0], [5, 40, 6]
                if name == "padded positions":
                    prefix, extend = [0, 0], [4, 4]
                fb, batch = _extend(prefix, extend, **kwargs)
                self.assertIsNone(
                    fb._text_only_mrope_positions(batch=batch, rl_on_policy_target=None)
                )
        mm = [None, _mm(40), None]
        self.assert_same(
            _rows([0, 0, 0], [5, 40, 6], mm=mm),
            _rows([0, 0, 0], [5, 40, 6], mm=mm, host_loop=True),
        )

    @unittest.skipUnless(_HAS_CUDA, "needs a GPU")
    def test_triton_positions_equal_the_host_loop_gpu(self):
        prefix, extend = [1000, 0, 4096], [32768, 512, 64]
        positions = fbi.compute_position(
            "triton",
            torch.tensor(prefix, dtype=torch.int32, device="cuda"),
            torch.tensor(extend, dtype=torch.int32, device="cuda"),
            sum(extend),
        )[0]
        want = _rows(prefix, extend, device="cuda", host_loop=True)
        got = _rows(prefix, extend, device="cuda", positions=positions)
        self.assert_same(got, want)


if __name__ == "__main__":
    unittest.main()
