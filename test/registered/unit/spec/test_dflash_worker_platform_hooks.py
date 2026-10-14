"""DFlash V2 platform hooks: portable defaults, ROCm worker selection and the
ROCm worker's fallbacks to the portable path."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import spec_info
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dflash_worker_v2_rocm import DFlashWorkerV2ROCm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TAIL_KW = dict(
    target_hidden=torch.zeros(8, 4),
    cache_loc=torch.zeros(8, dtype=torch.int64),
    cache_loc_2d=torch.zeros(1, 8, dtype=torch.int64),
    positions=torch.zeros(8, dtype=torch.int64),
    commit_lens=torch.ones(1, dtype=torch.int32),
    bs=1,
)


def _worker(cls, **attrs):
    worker = object.__new__(cls)
    worker.__dict__.update(attrs)
    return worker


def _mamba_kw():
    return dict(
        batch=SimpleNamespace(mamba_track_indices=None),
        commit_lens=torch.tensor([1, 3, 8], dtype=torch.int32),
        seq_lens_pre_verify=torch.tensor([10, 20, 30]),
        seq_lens_post_verify=torch.tensor([11, 23, 38]),
    )


class TestDFlashWorkerPlatformHooks(CustomTestCase):
    def test_create_worker_selects_rocm_worker_on_hip(self):
        for hip, expected in ((True, DFlashWorkerV2ROCm), (False, DFlashWorkerV2)):
            with (
                self.subTest(hip=hip),
                mock.patch("sglang.srt.utils.is_hip", return_value=hip),
                mock.patch.object(spec_info, "resolving_view", lambda args: args),
            ):
                worker_cls = spec_info.SpeculativeAlgorithm.DFLASH.create_worker(
                    SimpleNamespace()
                )
                self.assertIs(worker_cls, expected)

    def test_portable_defaults(self):
        worker = _worker(DFlashWorkerV2)
        self.assertFalse(DFlashWorkerV2._packed_draft_argmax)
        self.assertFalse(DFlashWorkerV2._publish_before_mamba_commit)
        logits = torch.randn(6, 33)
        self.assertTrue(
            torch.equal(worker._verify_argmax(logits), torch.argmax(logits, dim=-1))
        )
        self.assertFalse(worker._kv_fuse_store(None, None, None, None, None, None))
        self.assertFalse(worker._replay_tail_graph(**_TAIL_KW))
        steps, track = worker._mamba_commit_steps(**_mamba_kw())
        self.assertTrue(torch.equal(steps, torch.tensor([0, 2, 7])))
        self.assertEqual(steps.dtype, torch.int64)
        self.assertIsNone(track)

    def test_rocm_worker_falls_back_off_device(self):
        attrs = dict(
            device=torch.device("cpu"),
            draft_window_size=8,
            page_size=1,
            _use_triton_compact_rebuild=True,
            _tail_graph_on=False,
        )
        rocm = _worker(DFlashWorkerV2ROCm, **attrs)
        portable = _worker(DFlashWorkerV2, **attrs)
        self.assertTrue(DFlashWorkerV2ROCm._packed_draft_argmax)
        self.assertTrue(DFlashWorkerV2ROCm._publish_before_mamba_commit)

        logits = torch.randn(6, 33)
        self.assertTrue(
            torch.equal(rocm._verify_argmax(logits), portable._verify_argmax(logits))
        )
        for got, ref in zip(
            rocm._mamba_commit_steps(**_mamba_kw()),
            portable._mamba_commit_steps(**_mamba_kw()),
        ):
            self.assertTrue(got is ref is None or torch.equal(got, ref))
        prefix_lens = torch.tensor([3, 8, 20], dtype=torch.int32)
        got, got_suffix = rocm._compact_draft_prefix_lens(prefix_lens)
        ref, ref_suffix = portable._compact_draft_prefix_lens(prefix_lens)
        self.assertTrue(torch.equal(got, ref))
        self.assertIsNone(got_suffix)
        self.assertIsNone(ref_suffix)
        # Disabled tail graph: the caller runs the eager append.
        self.assertFalse(rocm._replay_tail_graph(**_TAIL_KW))


if __name__ == "__main__":
    unittest.main()
