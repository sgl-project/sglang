"""Unit tests for DFLASH + mamba-radix-cache-strategy extra_buffer_lazy:
server_args validation accepts the pairing, and DFlashVerifyInput.prepare_for_verify
runs prepare_mamba_track_for_verify (the hook eagle/ngram/dspark already run)
before ForwardBatch.init_new snapshots the track fields."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.mamba_hook import validate_mamba_extra_buffer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import dflash_info
from sglang.srt.speculative.dflash_info import DFlashVerifyInput


def _lazy_view(**overrides):
    view = SimpleNamespace(
        mamba_radix_cache_strategy="extra_buffer_lazy",
        disaggregation_mode="null",
        speculative_algorithm="DFLASH",
        speculative_num_draft_tokens=8,
        mamba_track_interval=256,
        page_size=64,
        chunked_prefill_size=None,
    )
    for key, value in overrides.items():
        setattr(view, key, value)
    return view


class TestValidateMambaExtraBufferLazyDflash(CustomTestCase):
    """The DFLASH rejection is gone; the neighboring invariants still hold."""

    def _validate(self, view):
        with mock.patch(
            "sglang.srt.arg_groups.overrides.supports_mamba_cache_extra_buffer",
            return_value=True,
        ), mock.patch(
            # Keep the test runnable on CPU-only hosts: the platform assert is
            # not what is under test here.
            "sglang.srt.arg_groups.mamba_hook.is_cuda",
            return_value=True,
        ):
            validate_mamba_extra_buffer(
                view,
                "Qwen3NextForCausalLM",
                mamba_cache_chunk_size_of=lambda: 64,
            )

    def test_dflash_with_extra_buffer_lazy_is_accepted(self):
        self._validate(_lazy_view())

    def test_dspark_still_accepted(self):
        self._validate(_lazy_view(speculative_algorithm="DSPARK"))

    def test_pd_disaggregation_still_rejected(self):
        with self.assertRaisesRegex(AssertionError, "PD disaggregation"):
            self._validate(_lazy_view(disaggregation_mode="decode"))

    def test_track_interval_must_cover_draft_tokens(self):
        with self.assertRaises(AssertionError):
            self._validate(
                _lazy_view(speculative_num_draft_tokens=512, mamba_track_interval=256)
            )

    def test_the_chunk_size_is_not_read_before_the_page_size_resolves(self):
        """`mamba_cache_chunk_size` is derived from `page_size`, which the
        pipeline writes *after* `_handle_model_specific_adjustments` runs this
        validator. The read has to stay inside the `page_size is not None`
        guard: evaluating it at the call site raises `TypeError` on the
        unresolved `None` (hit by Qwen3-Next under PD disaggregation)."""
        from sglang.srt.arg_groups.mamba_hook import validate_mamba_extra_buffer

        def _must_not_be_read():
            raise AssertionError("the chunk size was read before page_size resolved")

        with mock.patch(
            "sglang.srt.arg_groups.overrides.supports_mamba_cache_extra_buffer",
            return_value=True,
        ), mock.patch("sglang.srt.arg_groups.mamba_hook.is_cuda", return_value=True):
            validate_mamba_extra_buffer(
                _lazy_view(page_size=None),
                "Qwen3NextForCausalLM",
                mamba_cache_chunk_size_of=_must_not_be_read,
            )


class TestDflashVerifyRunsMambaTrackHook(CustomTestCase):
    """prepare_for_verify calls prepare_mamba_track_for_verify after the batch
    is stamped TARGET_VERIFY and before ForwardBatch.init_new; idle batches
    skip the hook."""

    def _spec_input(self):
        return DFlashVerifyInput(
            draft_token=torch.tensor([1, 2, 3, 4], dtype=torch.long),
            positions=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            draft_token_num=4,
        )

    def _run(self, forward_mode):
        calls = []
        batch = SimpleNamespace(forward_mode=forward_mode)
        attn_backend = SimpleNamespace(
            init_forward_metadata=lambda fb: calls.append("init_forward_metadata")
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                decode_cuda_graph_runner=None, attn_backend=attn_backend
            )
        )

        def fake_hook(hook_batch):
            calls.append(("hook", hook_batch.forward_mode))

        fake_forward_batch = SimpleNamespace()

        def fake_init_new(*args, **kwargs):
            calls.append("init_new")
            return fake_forward_batch

        with mock.patch(
            "sglang.srt.speculative.spec_utils.prepare_mamba_track_for_verify",
            side_effect=fake_hook,
        ), mock.patch.object(
            dflash_info.ForwardBatch, "init_new", side_effect=fake_init_new
        ):
            out, can_run_cuda_graph = self._spec_input().prepare_for_verify(
                batch, target_worker
            )
        self.assertIs(out, fake_forward_batch)
        self.assertFalse(can_run_cuda_graph)
        return calls, batch

    def test_hook_runs_before_init_new_on_verify(self):
        calls, batch = self._run(ForwardMode.DECODE)
        self.assertEqual(
            calls,
            [("hook", ForwardMode.TARGET_VERIFY), "init_new", "init_forward_metadata"],
        )
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)

    def test_idle_batch_skips_hook(self):
        calls, batch = self._run(ForwardMode.IDLE)
        self.assertEqual(calls, ["init_new"])
        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)


if __name__ == "__main__":
    unittest.main()
