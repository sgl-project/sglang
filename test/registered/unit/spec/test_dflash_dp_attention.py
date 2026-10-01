import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.parallel_hook import handle_deprecated_dp_attention
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(device="cuda", **overrides):
    return ServerArgs(
        model_path="dummy",
        device=device,
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        speculative_num_draft_tokens=4,
        speculative_draft_attention_backend="triton",
        **overrides,
    )


class TestDFlashDPAttention(CustomTestCase):
    def _resolve(self, args, architecture="DFlashDraftModel"):
        # Only checkpoint IO and host-platform discovery are substituted. The
        # real algorithm dispatcher and resolution stash enforce the gate.
        with (
            patch(
                "sglang.srt.utils.hf_transformers_utils.get_config",
                return_value=SimpleNamespace(architectures=[architecture]),
            ),
            patch(
                "sglang.srt.arg_groups.speculative_hook.current_platform.is_out_of_tree",
                return_value=False,
            ),
        ):
            handle_speculative_decoding(args)

    def test_cuda_and_npu_dp_attention_require_local_lm_head(self):
        for device in ("cuda", "cuda:0", "npu"):
            with self.subTest(device=device):
                args = _args(device, tp_size=2, attn_dp_size=2)
                with self.assertRaisesRegex(ValueError, "--enable-dp-lm-head"):
                    self._resolve(args)

    def test_cuda_and_npu_dp_attention_with_local_head_resolve(self):
        for device in ("cuda", "cuda:0", "npu"):
            with self.subTest(device=device):
                args = _args(device, tp_size=2, attn_dp_size=2, enable_dp_lm_head=True)
                self._resolve(args)
                self.assertEqual(resolution_result(args, "attn_dp_size"), 2)
                self.assertEqual(resolution_result(args, "speculative_num_steps"), 1)
                self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 1)
                self.assertEqual(
                    resolution_result(args, "speculative_num_draft_tokens"), 4
                )

    def test_xpu_dp_attention_remains_unsupported_with_local_head(self):
        for local_head in (False, True):
            with self.subTest(local_head=local_head):
                args = _args(
                    "xpu", tp_size=2, attn_dp_size=2, enable_dp_lm_head=local_head
                )
                with self.assertRaisesRegex(ValueError, "dp attention.*CUDA.*NPU"):
                    self._resolve(args)

    def test_degenerate_dp_does_not_require_local_head(self):
        for device in ("cuda", "npu", "xpu"):
            for legacy_flag in (False, True):
                with self.subTest(device=device, legacy_flag=legacy_flag):
                    args = _args(device, dp_size=1, enable_dp_attention=legacy_flag)
                    handle_deprecated_dp_attention(args)
                    self._resolve(args)
                    self.assertEqual(resolution_result(args, "attn_dp_size"), 1)
                    self.assertFalse(resolution_result(args, "enable_dp_lm_head"))

    def test_legacy_dp_spelling_uses_resolved_attention_dp_size(self):
        args = _args(tp_size=2, dp_size=2, enable_dp_attention=True)
        handle_deprecated_dp_attention(args)
        # The raw record is still 1; the DFlash gate must use the resolved 2.
        self.assertEqual(args.attn_dp_size, 1)
        self.assertEqual(resolution_result(args, "attn_dp_size"), 2)
        with self.assertRaisesRegex(ValueError, "--enable-dp-lm-head"):
            self._resolve(args)

    def test_both_draft_architectures_keep_shared_worker_route(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        for architecture in ("DFlashDraftModel", "DFlash2DraftModel"):
            for disable_overlap in (False, True):
                with self.subTest(
                    architecture=architecture, disable_overlap=disable_overlap
                ):
                    args = _args(
                        tp_size=2,
                        attn_dp_size=2,
                        enable_dp_lm_head=True,
                        disable_overlap_schedule=disable_overlap,
                    )
                    self._resolve(args, architecture)
                    algorithm = SpeculativeAlgorithm.from_string(
                        resolution_result(args, "speculative_algorithm")
                    )
                    self.assertIs(algorithm, SpeculativeAlgorithm.DFLASH)
                    self.assertIs(algorithm.create_worker(args), DFlashWorkerV2)

    def test_dp_lm_head_does_not_bypass_pipeline_rejection(self):
        args = _args(tp_size=2, attn_dp_size=2, enable_dp_lm_head=True, pp_size=2)
        with self.assertRaisesRegex(ValueError, "pp_size == 1"):
            self._resolve(args)

    def test_dp_attention_rejects_context_parallel(self):
        args = _args(tp_size=4, attn_dp_size=2, attn_cp_size=2, enable_dp_lm_head=True)
        with self.assertRaisesRegex(ValueError, "does not support context parallel"):
            self._resolve(args)

    def test_dp_draft_stays_eager_without_building_graph_sampler(self):
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.speculative import dflash_worker_v2 as worker_module

        for attention_dp in (False, True):
            with self.subTest(attention_dp=attention_dp):
                worker = SimpleNamespace(
                    draft_owns_attention=True,
                    _target_tp_rank=0,
                    _draft_worker=Mock(),
                    _maybe_build_draft_sampler=Mock(return_value=None),
                )
                execution = SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=SimpleNamespace(
                            decode=SimpleNamespace(backend=Backend.FULL)
                        )
                    )
                )
                with (
                    patch.object(worker_module, "draft_pp_context", nullcontext),
                    patch.object(
                        worker_module, "draft_tp_context", lambda _: nullcontext()
                    ),
                    patch.object(worker_module, "get_exec", return_value=execution),
                    patch.object(
                        worker_module,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_dp_enabled=attention_dp),
                    ),
                    patch.object(worker_module, "is_cuda", return_value=False),
                    patch.object(
                        worker_module.current_platform,
                        "is_out_of_tree",
                        return_value=False,
                    ),
                ):
                    worker_module.DFlashWorkerV2.init_cuda_graphs(worker)
                worker._draft_worker.init_cuda_graphs.assert_called_once_with(
                    capture_decode_cuda_graph=not attention_dp
                )
                if attention_dp:
                    worker._maybe_build_draft_sampler.assert_not_called()
                else:
                    worker._maybe_build_draft_sampler.assert_called_once_with()


class TestDFlashVerifyPlanning(CustomTestCase):
    def _prepare(self, *, idle=False, force_eager=False, graph=True, npu=False):
        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative import dflash_info

        verify_input = dflash_info.DFlashVerifyInput(
            draft_token=torch.arange(4),
            positions=torch.arange(4),
            draft_token_num=4,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE if idle else ForwardMode.DECODE
        )
        forward = SimpleNamespace(can_run_decode_cuda_graph=True)
        runner = Mock()
        runner.can_run_graph.return_value = True
        backend = Mock()
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                decode_cuda_graph_runner=runner if graph else None,
                attn_backend=backend,
            )
        )
        with (
            patch.object(dflash_info, "_is_npu", npu),
            patch.object(dflash_info.ForwardBatch, "init_new", return_value=forward),
            patch("sglang.srt.speculative.spec_utils.prepare_mamba_track_for_verify"),
        ):
            result = verify_input.prepare_for_verify(
                batch, worker, force_eager=force_eager
            )
        return result, runner, backend, verify_input

    def test_forced_eager_skips_graph_and_attention_preplanning(self):
        # In particular, an idle rank must not enter graph load_batch, which
        # overwrites its causal DFlash mask with the generic graph mask.
        for idle in (False, True):
            with self.subTest(idle=idle):
                (forward, can_graph), runner, backend, verify_input = self._prepare(
                    idle=idle, force_eager=True
                )
                self.assertFalse(can_graph)
                self.assertFalse(forward.can_run_decode_cuda_graph)
                self.assertIsNone(verify_input.custom_mask)
                runner.can_run_graph.assert_not_called()
                runner.load_batch.assert_not_called()
                backend.init_forward_metadata.assert_not_called()

    def test_default_verify_keeps_graph_planning(self):
        # Fully busy DFlash and shared DSpark callers retain the existing path.
        (forward, can_graph), runner, backend, _ = self._prepare()
        self.assertTrue(can_graph)
        runner.can_run_graph.assert_called_once_with(forward)
        runner.load_batch.assert_called_once_with(forward)
        backend.init_forward_metadata.assert_not_called()

    def test_default_without_graph_keeps_eager_preplanning(self):
        (forward, can_graph), _, backend, _ = self._prepare(graph=False)
        self.assertFalse(can_graph)
        backend.init_forward_metadata.assert_called_once_with(forward)

    def test_npu_idle_keeps_deferred_planning(self):
        for force_eager in (False, True):
            with self.subTest(force_eager=force_eager):
                (_, can_graph), runner, backend, _ = self._prepare(
                    idle=True, npu=True, force_eager=force_eager
                )
                self.assertEqual(can_graph, not force_eager)
                runner.load_batch.assert_not_called()
                backend.init_forward_metadata.assert_not_called()


if __name__ == "__main__":
    unittest.main()
