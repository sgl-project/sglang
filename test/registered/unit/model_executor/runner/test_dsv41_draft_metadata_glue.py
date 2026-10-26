"""The DSpark draft runner's metadata prep through the glue graph. Only a backend that declares its prep host-free accepts it
(the DSV4 backend, then taking the draft block's extend lengths from device
constants), and the runner hands the graph the registry's out_cache_loc slot instead
of the step's own tensor."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend(**overrides):
    backend = dsv4.DeepseekV4AttnBackend.__new__(dsv4.DeepseekV4AttnBackend)
    fields = dict(
        is_dspark_draft=True,
        trtllm_attn=False,
        online_c128_mtp=SimpleNamespace(enabled=lambda: False),
        token_to_kv_pool=SimpleNamespace(request_window=None),
        draft_metadata_glued=False,
        _draft_extend_lens={},
    )
    fields.update(overrides)
    for name, value in fields.items():
        setattr(backend, name, value)
    return backend


def _parallel(cp=1):
    return mock.patch.object(
        dsv4, "get_parallel", return_value=SimpleNamespace(attn_cp_size=cp)
    )


class TestHostFreeDraftPrep(CustomTestCase):
    def test_the_audited_setup_accepts_and_glues(self):
        backend = _backend()
        with _parallel():
            self.assertEqual(backend.glue_draft_metadata(), "")
        self.assertTrue(backend.draft_metadata_glued)

    def test_every_other_setup_says_why(self):
        cases = {
            "not the DSpark draft runner": dict(is_dspark_draft=False),
            "trtllm-gen attention": dict(trtllm_attn=True),
            "online c128 MTP": dict(
                online_c128_mtp=SimpleNamespace(enabled=lambda: True)
            ),
            "a request window": dict(
                token_to_kv_pool=SimpleNamespace(request_window=object())
            ),
        }
        for why, overrides in cases.items():
            backend = _backend(**overrides)
            with _parallel(), self.subTest(why=why):
                self.assertEqual(backend.glue_draft_metadata(), why)
                self.assertFalse(backend.draft_metadata_glued)
        with _parallel(cp=2):
            self.assertEqual(_backend().glue_draft_metadata(), "context parallelism")


class TestDraftExtendLengths(CustomTestCase):
    def _moving(self, backend):
        moved = []

        def move(values):
            moved.append(list(values))
            return torch.tensor(values, dtype=torch.int32)

        backend._move_to_device = move
        return moved

    def test_glued_keeps_one_constant_per_batch_size(self):
        backend = _backend(draft_metadata_glued=True)
        moved = self._moving(backend)
        first = backend._draft_block_extend_lens([6, 6])
        self.assertIs(backend._draft_block_extend_lens([6, 6]), first)
        backend._draft_block_extend_lens([6])
        self.assertEqual(moved, [[6, 6], [6]])
        self.assertEqual(first.tolist(), [6, 6])

    def test_glued_refuses_a_new_constant_inside_a_capture(self):
        backend = _backend(draft_metadata_glued=True)
        self._moving(backend)
        backend._draft_block_extend_lens([6])
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            # a warm-up made this one: the capture reads it by address
            backend._draft_block_extend_lens([6])
            with self.assertRaisesRegex(RuntimeError, "first seen inside"):
                backend._draft_block_extend_lens([6, 6])


class TestDraftBlockMetadata(CustomTestCase):
    def _backend(self):
        backend = _backend(
            draft_metadata_glued=True,
            page_size=256,
            has_c4=False,
            has_c128=False,
            req_to_token=torch.empty((4, 2048), dtype=torch.int32),
        )
        # Device allocation and the page/SWA kernels are the accelerator boundary.
        backend._move_to_device = lambda lens: torch.tensor(lens, dtype=torch.int32)
        backend.make_core_attn_metadata = lambda **kwargs: SimpleNamespace(**kwargs)
        return backend

    @staticmethod
    def _expand_on_cpu(**kwargs):
        # The device-only torch reference reads the same tensors as the CUDA kernel.
        lens = kwargs["extend_seq_lens"]
        kwargs["extend_start_loc"] = lens.cumsum(0) - lens
        return dsv4.ExpandPrefillCausally.torch(**kwargs)

    def test_missing_host_mirror_never_reads_back_and_uses_live_lengths(self):
        """Draft metadata capture must not read device lengths into a Python list,
        and later steps must use the updated lengths rather than warm-up values."""
        for prefixes, block_size in (([127], 6), ([127, 1023], 3)):
            with self.subTest(prefixes=prefixes, block_size=block_size):
                backend = self._backend()
                bs = len(prefixes)
                seq_lens = torch.tensor(prefixes, dtype=torch.int32)
                req_pool_indices = torch.arange(bs, dtype=torch.int32).flip(0)
                inputs = dict(
                    max_seq_len=2048,
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    seq_lens_cpu=None,
                    out_cache_loc=torch.arange(bs * block_size),
                    block_size=block_size,
                )
                with (
                    mock.patch.object(
                        torch.Tensor, "is_cuda", property(lambda _: True)
                    ),
                    mock.patch.object(
                        torch.Tensor, "tolist", side_effect=AssertionError("host read")
                    ),
                    mock.patch.object(
                        torch.Tensor, "item", side_effect=AssertionError("host read")
                    ),
                    mock.patch.object(
                        torch.Tensor, "cpu", side_effect=AssertionError("host read")
                    ),
                    mock.patch.object(
                        dsv4.ExpandPrefillCausally,
                        "triton",
                        side_effect=self._expand_on_cpu,
                    ),
                ):
                    first = backend.init_forward_metadata_dspark_draft_block(**inputs)
                    seq_lens.add_(11)
                    with (
                        mock.patch("torch.cuda.is_available", return_value=True),
                        mock.patch(
                            "torch.cuda.is_current_stream_capturing", return_value=True
                        ),
                    ):
                        second = backend.init_forward_metadata_dspark_draft_block(
                            **inputs
                        )
                expected = torch.tensor(
                    [p + step for p in prefixes for step in range(1, block_size + 1)],
                    dtype=torch.int32,
                )
                torch.testing.assert_close(
                    first.core_metadata.seq_lens_casual, expected
                )
                torch.testing.assert_close(
                    second.core_metadata.seq_lens_casual, expected + 11
                )
                torch.testing.assert_close(
                    second.core_metadata.req_pool_indices_repeated,
                    req_pool_indices.repeat_interleave(block_size),
                )

    def test_cpu_expansion_still_resolves_missing_or_provided_host_lengths(self):
        backend = self._backend()
        seq_lens = torch.tensor([127, 1023], dtype=torch.int32)
        for mirror in (None, seq_lens.clone()):
            with self.subTest(has_mirror=mirror is not None):
                metadata = backend.init_forward_metadata_dspark_draft_block(
                    max_seq_len=2048,
                    req_pool_indices=torch.tensor([3, 1], dtype=torch.int32),
                    seq_lens=seq_lens,
                    seq_lens_cpu=mirror,
                    out_cache_loc=torch.arange(6),
                    block_size=3,
                )
                torch.testing.assert_close(
                    metadata.core_metadata.seq_lens_casual,
                    torch.tensor([128, 129, 130, 1024, 1025, 1026], dtype=torch.int32),
                )


def _runner(*, draft=True, backend=None):
    runner = mod.DecodeCudaGraphRunner.__new__(mod.DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(is_draft_worker=draft)
    runner.attn_backend = backend if backend is not None else _backend()
    return runner


class TestRunnerGluesTheDraft(CustomTestCase):
    def test_the_glue_reads_the_registry_slot(self):
        runner = _runner()
        runner.buffers = SimpleNamespace(out_cache_loc=torch.arange(16))
        step_own = torch.arange(100, 106)
        view = SimpleNamespace(out_cache_loc=step_own)
        runner._draft_metadata_glue = False
        runner._glue_draft_inputs(view)
        self.assertIs(view.out_cache_loc, step_own)
        runner._draft_metadata_glue = True
        runner._glue_draft_inputs(view)
        slot = view.out_cache_loc
        self.assertEqual(slot.data_ptr(), runner.buffers.out_cache_loc.data_ptr())
        self.assertEqual(slot.shape, step_own.shape)
        idle = SimpleNamespace(out_cache_loc=None)
        runner._glue_draft_inputs(idle)
        self.assertIsNone(idle.out_cache_loc)


if __name__ == "__main__":
    unittest.main()
