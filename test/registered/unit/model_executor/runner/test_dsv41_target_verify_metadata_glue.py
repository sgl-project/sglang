"""The DSpark target verify runner's metadata prep through the glue graph: only behind
its switch, only for a static prep the DSV4 backend declares host-free, and reading the
registry's out_cache_loc slot like the draft runner's glue."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend(**overrides):
    backend = dsv4.DeepseekV4AttnBackend.__new__(dsv4.DeepseekV4AttnBackend)
    fields = dict(
        is_draft_runner=False,
        needs_cpu_seq_lens=False,
        trtllm_attn=False,
        online_c128_mtp=SimpleNamespace(enabled=lambda: False),
        token_to_kv_pool=SimpleNamespace(request_window=None),
    )
    fields.update(overrides)
    for name, value in fields.items():
        setattr(backend, name, value)
    return backend


def _parallel(cp=1):
    return mock.patch.object(
        dsv4, "get_parallel", return_value=SimpleNamespace(attn_cp_size=cp)
    )


class TestHostFreeTargetVerifyPrep(CustomTestCase):
    def test_every_other_setup_says_why(self):
        cases = (
            (dict(is_draft_runner=True), "a draft runner"),
            (dict(needs_cpu_seq_lens=True), "host sequence lengths"),
            (dict(trtllm_attn=True), "trtllm-gen attention"),
            (
                dict(online_c128_mtp=SimpleNamespace(enabled=lambda: True)),
                "online c128 MTP",
            ),
            (
                dict(token_to_kv_pool=SimpleNamespace(request_window=object())),
                "a request window",
            ),
        )
        with _parallel():
            for overrides, why in cases:
                with self.subTest(why=why):
                    self.assertEqual(
                        _backend(**overrides).glue_target_verify_metadata(), why
                    )
            with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
                self.assertEqual(
                    _backend().glue_target_verify_metadata(), "ragged verify"
                )
        with _parallel(cp=2):
            self.assertEqual(
                _backend().glue_target_verify_metadata(), "context parallelism"
            )


def _runner(*, draft=False, dspark=True, mode=ForwardMode.TARGET_VERIFY, backend=None):
    runner = mod.DecodeCudaGraphRunner.__new__(mod.DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        is_draft_worker=draft,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: dspark),
    )
    runner.capture_forward_mode = mode
    runner.attn_backend = backend if backend is not None else _backend()
    return runner


class TestRunnerGluesTheTargetVerify(CustomTestCase):
    def test_on_only_behind_the_switch_for_a_host_free_dspark_verify(self):
        with _parallel():
            self.assertFalse(_runner()._glues_target_verify_metadata())
            with envs.SGLANG_OPT_DSV41_TARGET_VERIFY_METADATA_GLUE.override(True):
                self.assertTrue(_runner()._glues_target_verify_metadata())
                for runner in (
                    _runner(draft=True),
                    _runner(dspark=False),
                    _runner(mode=ForwardMode.DECODE),
                    _runner(backend=_backend(trtllm_attn=True)),
                ):
                    self.assertFalse(runner._glues_target_verify_metadata())

    def test_the_target_glue_reads_the_registry_slot(self):
        runner = _runner()
        runner.buffers = SimpleNamespace(out_cache_loc=torch.arange(16))
        view = SimpleNamespace(out_cache_loc=torch.arange(100, 106))
        runner._draft_metadata_glue, runner._target_metadata_glue = False, True
        runner._glue_draft_inputs(view)
        self.assertEqual(
            view.out_cache_loc.data_ptr(), runner.buffers.out_cache_loc.data_ptr()
        )
        self.assertEqual(view.out_cache_loc.shape, (6,))


if __name__ == "__main__":
    unittest.main()
