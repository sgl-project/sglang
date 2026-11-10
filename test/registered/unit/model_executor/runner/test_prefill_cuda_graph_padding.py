import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.managers.scheduler_components.dp_attn import (
    _local_prefill_cuda_graph_vote,
)
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    prefill_graph_tolerates_sum_len,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _requires_exact_gdn_prefill_bucket,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_context, override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.prefer_eager_mixed_prefill = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_context_size = None
        runner.max_seq_len = None
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens, mode=ForwardMode.EXTEND):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            contains_mm_inputs=lambda: False,
            forward_mode=mode,
            global_forward_mode=None,
            _original_forward_mode=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
            seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int64),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int64),
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_mixed_batch_uses_scoped_runner_policy(self):
        runner = self._make_runner()
        batch = self._make_forward_batch(8, mode=ForwardMode.MIXED)

        self.assertTrue(runner.can_run_graph(batch))
        runner.prefer_eager_mixed_prefill = True
        self.assertFalse(runner.can_run_graph(batch))

    def test_replay_without_captured_metadata_builds_snapshot(self):
        runner = self._make_runner()
        runner.use_captured_attn_metadata = False
        attn_backend = mock.Mock()
        attn_backend.resolve_prefill_shared_read_ends.return_value = mock.sentinel.end
        runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
        forward_batch = self._make_forward_batch(8)
        static_forward_batch = self._make_forward_batch(16)

        shared_read_ends = runner._prepare_forward_metadata_for_replay(
            forward_batch,
            static_forward_batch,
            shape_key=ShapeKey(size=16),
        )

        self.assertIs(shared_read_ends, mock.sentinel.end)
        attn_backend.init_forward_metadata.assert_called_once_with(forward_batch)
        attn_backend.resolve_prefill_shared_read_ends.assert_called_once_with(
            forward_batch, build_snapshot=True
        )

    def test_full_replay_pads_request_slot_cpu_mirror(self):
        for batch_size, has_mirror in ((2, True), (4, True), (2, False)):
            with self.subTest(batch_size=batch_size, has_mirror=has_mirror):
                runner = self._make_runner()
                runner._is_full_backend = True
                runner._capture_req_slots = 4
                runner._full_cg_seq_lens_cpu = torch.full((4,), -1)
                slots = torch.tensor([7, 2, 9, 5])[:batch_size]
                seq_lens = torch.arange(1, batch_size + 1)
                static_slots = torch.zeros(4, dtype=slots.dtype)
                static_slots[:batch_size].copy_(slots)
                static_lens = torch.zeros(4, dtype=seq_lens.dtype)
                static_lens[:batch_size].copy_(seq_lens)
                runner._prefill_static_buffers = {
                    "req_pool_indices": static_slots,
                    "seq_lens": static_lens,
                    "extend_seq_lens": static_lens.clone(),
                    "extend_prefix_lens": torch.zeros(4, dtype=torch.int64),
                }
                attn_backend = mock.Mock()
                runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
                batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=batch_size,
                    input_ids=torch.arange(batch_size),
                    req_pool_indices=slots,
                    req_pool_indices_cpu=slots if has_mirror else None,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens,
                    out_cache_loc=torch.arange(batch_size),
                    seq_lens_sum=int(seq_lens.sum()),
                )

                static_batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=4,
                    input_ids=torch.arange(16),
                    req_pool_indices=static_slots,
                    seq_lens=static_lens,
                    seq_lens_cpu=static_lens,
                    out_cache_loc=torch.arange(16),
                    seq_lens_sum=int(static_lens.sum()),
                )

                runner._prepare_forward_metadata_for_replay(
                    batch, static_batch, shape_key=ShapeKey(size=16)
                )

                attn_backend.init_forward_metadata_out_graph.assert_called_once()
                padded = attn_backend.init_forward_metadata_out_graph.call_args.args[0]
                self.assertIsInstance(padded, ForwardBatch)
                self.assertIsNot(padded, batch)
                self.assertEqual(padded.batch_size, 4)
                self.assertIs(padded.input_ids, static_batch.input_ids)
                torch.testing.assert_close(padded.seq_lens_cpu, static_lens)
                torch.testing.assert_close(padded.req_pool_indices, static_slots)
                if has_mirror:
                    torch.testing.assert_close(
                        padded.req_pool_indices_cpu, static_slots
                    )
                    self.assertEqual(padded.req_pool_indices_cpu.device.type, "cpu")
                    self.assertIs(batch.req_pool_indices_cpu, slots)
                else:
                    self.assertIsNone(padded.req_pool_indices_cpu)
                self.assertEqual(batch.batch_size, batch_size)
                self.assertEqual(batch.input_ids.numel(), batch_size)
                torch.testing.assert_close(batch.seq_lens_cpu, seq_lens)
                attn_backend.init_forward_metadata.assert_not_called()

    def _megamoe_no_prefill_cp(self, graph_has_dp_gather=False):
        return (
            mock.patch.object(
                forward_batch_info,
                "get_flags",
                return_value=SimpleNamespace(
                    dp=SimpleNamespace(prefill_graph_has_dp_gather=graph_has_dp_gather)
                ),
            ),
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MEGAMOE,
            ),
            mock.patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.layers.cp.utils.is_mla_cp_enabled",
                return_value=False,
            ),
        )

    def test_megamoe_idle_rank_without_graph_gather_keeps_sum_len(self):
        # Without a DP gather in the graph, an eager idle rank matches the
        # replaying peers, so the sparse batch keeps per-rank buckets.
        runner = self._make_runner()
        flags, a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp()
        with flags, a2a, dsa_cp, mla_cp:
            self.assertTrue(prefill_graph_tolerates_sum_len())
            self.assertFalse(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 0])
                )
            )

    def test_megamoe_all_ranks_busy_keeps_per_rank_buckets(self):
        runner = self._make_runner()
        flags, a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp()
        with flags, a2a, dsa_cp, mla_cp:
            self.assertTrue(prefill_graph_tolerates_sum_len())
            self.assertFalse(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 16])
                )
            )

    def test_megamoe_graph_with_dp_gather_forces_shared_bucket(self):
        # A DP gather captured in the graph has fixed MAX_LEN geometry; per-rank
        # buckets or an eager idle rank would deadlock its all_gather.
        runner = self._make_runner()
        flags, a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp(
            graph_has_dp_gather=True
        )
        with flags, a2a, dsa_cp, mla_cp:
            self.assertFalse(prefill_graph_tolerates_sum_len())
            self.assertTrue(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 0])
                )
            )

    def test_rejects_context_above_fixed_maximum(self):
        runner = self._make_runner()
        runner.max_context_size = 700

        much_shorter = self._make_forward_batch(4)
        much_shorter.seq_lens_cpu.fill_(200)
        self.assertTrue(runner.can_run_graph(much_shorter))

        uncovered = self._make_forward_batch(4)
        uncovered.seq_lens_cpu.fill_(701)
        self.assertFalse(runner.can_run_graph(uncovered))

    def test_sequence_bound_votes_eager_for_active_and_idle_dp_ranks(self):
        runner = self._make_runner()
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.max_seq_len = 16384
        for context_len, expected in ((16384, True), (17000, False)):
            with self.subTest(context_len=context_len):
                active = self._make_forward_batch(4)
                active.seq_lens_cpu.fill_(context_len)
                schedule_batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=lambda: 1,
                    extend_num_tokens=4,
                    input_embeds=None,
                    replace_embeds=None,
                    prefix_lens=[context_len - 4],
                    return_logprob=False,
                    multimodal_inputs=None,
                    seq_lens_cpu=active.seq_lens_cpu,
                )
                votes = [
                    _local_prefill_cuda_graph_vote(
                        local_batch=batch,
                        num_tokens=4,
                        prefill_graph_runner=runner,
                        coordinated_prefill=True,
                        breakable_prefill=True,
                        spec_algorithm=None,
                        model_config=None,
                    )
                    for batch in (schedule_batch, None)
                ]
                self.assertEqual(votes, [expected, True])
                # The padded idle request fits; it must still honor the vote.
                idle = self._make_forward_batch(4)
                for batch in (active, idle):
                    batch.global_num_tokens_cpu = [4, 4]
                    batch.can_run_dp_prefill_cuda_graph = all(votes)
                    self.assertEqual(runner.can_run_graph(batch), expected)

    def test_context_and_sequence_bounds_both_apply(self):
        runner = self._make_runner()
        for context_limit, sequence_limit in ((700, 900), (900, 700)):
            with self.subTest(context=context_limit, sequence=sequence_limit):
                runner.max_context_size = context_limit
                runner.max_seq_len = sequence_limit
                batch = self._make_forward_batch(4)
                batch.seq_lens_cpu.fill_(700)
                self.assertTrue(runner.can_run_graph(batch))
                batch.seq_lens_cpu.fill_(701)
                self.assertFalse(runner.can_run_graph(batch))

    def test_unsupported_path_ignores_max_context_size(self):
        runner = self._make_runner()
        runner.max_context_size = 1024

        with self.assertLogs(runner_module.logger, level="WARNING") as logs:
            runner._ignore_max_context_size("test path")

        self.assertIsNone(runner.max_context_size)
        self.assertIn("fixed metadata extent", "\n".join(logs.output))

    def test_requires_exact_gdn_prefill_bucket_policy(self):
        """GDN prefill CUDA graphs allow padded replay by default only on SM100
        quantized FP8/FP4 models; non-SM100 or unquantized GDN models require
        exact bucket hits unless explicitly opted in or mixed-chunk is active."""
        gdn_leaf = GDNAttnBackend.__new__(GDNAttnBackend)
        hybrid_gdn = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        hybrid_gdn.linear_attn_backend = gdn_leaf
        hybrid_non_gdn = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        hybrid_non_gdn.linear_attn_backend = SimpleNamespace()
        non_gdn = SimpleNamespace()

        cases = (
            (hybrid_gdn, True, "modelopt", False, False, False),
            (hybrid_gdn, True, "modelopt_fp4", False, False, False),
            (hybrid_gdn, True, "modelopt_fp8", False, False, False),
            (hybrid_gdn, True, "modelopt_mixed", False, False, False),
            (hybrid_gdn, True, "fp8", False, False, False),
            (gdn_leaf, True, "modelopt_fp4", False, False, False),
            # Non-SM100 (e.g. SM120 / SM90) quantized GDN must require exact buckets.
            (hybrid_gdn, False, "modelopt_mixed", False, False, True),
            (hybrid_gdn, False, "modelopt_fp4", False, False, True),
            (hybrid_gdn, False, "fp8", False, False, True),
            # SM100 unquantized BF16 GDN must require exact buckets.
            (hybrid_gdn, True, None, False, False, True),
            (hybrid_gdn, True, "awq", False, False, True),
            # Non-GDN attention backends keep normal bucket padding.
            (non_gdn, False, None, False, False, False),
            (hybrid_non_gdn, False, "modelopt_mixed", False, False, False),
            # Explicit env opt-in or mixed-chunk prefill allows padded buckets.
            (hybrid_gdn, False, "modelopt_mixed", True, False, False),
            (hybrid_gdn, False, None, False, True, False),
        )
        for (
            attn_backend,
            is_sm100,
            quantization,
            allow_padded_env,
            mixed_chunk,
            expected,
        ) in cases:
            with (
                self.subTest(
                    backend=type(attn_backend).__name__,
                    is_sm100=is_sm100,
                    quantization=quantization,
                    allow_padded_env=allow_padded_env,
                    mixed_chunk=mixed_chunk,
                ),
                override_platform(is_sm100=is_sm100),
                envs.SGLANG_ENABLE_PADDED_GDN_PREFILL_GRAPH.override(allow_padded_env),
                get_context().override_server_args(enable_mixed_chunk=mixed_chunk),
            ):
                model_runner = SimpleNamespace(
                    attn_backend=attn_backend,
                    model_config=SimpleNamespace(quantization=quantization),
                )
                self.assertEqual(
                    _requires_exact_gdn_prefill_bucket(model_runner), expected
                )

    def test_exact_prefill_bucket_guard_and_dp_vote_consensus(self):
        """When exact GDN prefill buckets are required, padded token counts run
        eagerly while exact buckets replay, and active/idle DP ranks agree."""
        runner = self._make_runner()
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.capture_num_tokens = [4, 8, 12, 16, 20, 24, 28, 32]
        runner.max_num_tokens = 32
        runner._requires_exact_prefill_bucket = True

        padded_cases = (13, 14, 17, 21, 27, 29)
        exact_cases = (4, 16, 20, 24, 28, 32)

        for num_tokens in padded_cases:
            with self.subTest(local_only=num_tokens):
                self.assertFalse(
                    runner.can_run_graph(self._make_forward_batch(num_tokens))
                )
        for num_tokens in exact_cases:
            with self.subTest(local_only=num_tokens):
                self.assertTrue(
                    runner.can_run_graph(self._make_forward_batch(num_tokens))
                )

        for num_tokens, expected in [
            *((n, False) for n in padded_cases),
            *((n, True) for n in exact_cases),
        ]:
            with self.subTest(dp_tokens=num_tokens, expected=expected):
                active = self._make_forward_batch(num_tokens)
                schedule_batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=lambda: 1,
                    extend_num_tokens=num_tokens,
                    input_embeds=None,
                    replace_embeds=None,
                    prefix_lens=[0],
                    return_logprob=False,
                    multimodal_inputs=None,
                    seq_lens_cpu=active.seq_lens_cpu,
                )
                votes = [
                    _local_prefill_cuda_graph_vote(
                        local_batch=batch,
                        num_tokens=num_tokens if batch is not None else 0,
                        prefill_graph_runner=runner,
                        coordinated_prefill=True,
                        breakable_prefill=True,
                        spec_algorithm=None,
                        model_config=None,
                    )
                    for batch in (schedule_batch, None)
                ]
                self.assertEqual(votes, [expected, True])
                bucket = min(b for b in runner.capture_num_tokens if b >= num_tokens)
                idle = self._make_forward_batch(bucket)
                for batch in (active, idle):
                    batch.global_num_tokens_cpu = (
                        [bucket, bucket] if expected else [num_tokens, 0]
                    )
                    batch.can_run_dp_prefill_cuda_graph = all(votes)
                    self.assertEqual(runner.can_run_graph(batch), expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton GDN ba_proj")
class TestGDNBatchInvariantBaProjCUDA(CustomTestCase):
    def test_ba_proj_extend_is_batch_invariant_and_cuda_graph_safe(self):
        """EXTEND GDN ba_proj must keep real rows bit-identical across padded
        buckets and CUDA graph replay (default and side stream), while leaving
        DECODE and TARGET_VERIFY on the standard linear path."""
        import types

        import torch.nn as nn
        import torch.nn.functional as F

        import sglang.srt.models.qwen3_5 as qwen3_5_mod
        import sglang.srt.models.qwen3_next as qwen3_next_mod
        from sglang.srt.layers.linear import MergedColumnParallelLinear
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )
        from sglang.srt.utils import LazyValue

        device = torch.device("cuda")
        hidden_size = 5120
        num_v_heads = 48
        out_features = 2 * num_v_heads
        gen = torch.Generator(device=device).manual_seed(49133)

        ba_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            params_dtype=torch.bfloat16,
            parallel_group="replicated",
        ).to(device)
        ba_proj.weight.data.copy_(
            (
                torch.randn((out_features, hidden_size), generator=gen, device=device)
                * 0.02
            ).to(torch.bfloat16)
        )
        x_full = torch.randn(
            (128, hidden_size), generator=gen, device=device, dtype=torch.bfloat16
        )
        extend_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        mixed_batch = SimpleNamespace(forward_mode=ForwardMode.MIXED)
        decode_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        verify_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        for helper in (qwen3_5_mod._forward_ba_proj, qwen3_next_mod._forward_ba_proj):
            ref_13 = helper(ba_proj, x_full[:13].contiguous(), extend_batch)
            ref_fp32 = F.linear(x_full[:13].float(), ba_proj.weight.float()).to(
                torch.bfloat16
            )
            torch.testing.assert_close(ref_13, ref_fp32, atol=1e-2, rtol=1e-2)

            for padded_m in (16, 20, 24, 28, 32, 48, 64, 96, 128):
                x_padded = torch.zeros(
                    (padded_m, hidden_size), device=device, dtype=torch.bfloat16
                )
                x_padded[:13].copy_(x_full[:13])
                out_extend = helper(ba_proj, x_padded, extend_batch)
                self.assertEqual(tuple(out_extend.shape), (padded_m, out_features))
                self.assertTrue(
                    torch.equal(out_extend[:13], ref_13),
                    f"padded M={padded_m} diverged from M=13 on EXTEND",
                )
                out_mixed = helper(ba_proj, x_padded, mixed_batch)
                self.assertTrue(torch.equal(out_mixed[:13], ref_13))

            # Also verify 48 -> 96 and 64 -> 128 where Ampere cuBLAS F.linear splits.
            ref_48 = helper(ba_proj, x_full[:48].contiguous(), extend_batch)
            for padded_m in (64, 96, 128):
                x_padded = torch.zeros(
                    (padded_m, hidden_size), device=device, dtype=torch.bfloat16
                )
                x_padded[:48].copy_(x_full[:48])
                self.assertTrue(
                    torch.equal(
                        helper(ba_proj, x_padded, extend_batch)[:48],
                        ref_48,
                    )
                )
            # DECODE and TARGET_VERIFY stay on ba_proj.forward directly.
            for non_prefill_batch in (decode_batch, verify_batch, None):
                with mock.patch.object(
                    ba_proj, "forward", wraps=ba_proj.forward
                ) as spy_fwd:
                    out_non_prefill = helper(
                        ba_proj, x_full[:4].contiguous(), non_prefill_batch
                    )
                    spy_fwd.assert_called_once()
                expected_direct, _ = ba_proj(x_full[:4].contiguous())
                self.assertTrue(torch.equal(out_non_prefill, expected_direct))

        # Verify _forward_input_proj CUDA graph capture + replay on both
        # default stream and alt_stream for Qwen3.5 and Qwen3-Next GDN.
        qkvz_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[256, 256],
            bias=False,
            params_dtype=torch.bfloat16,
            parallel_group="replicated",
        ).to(device)
        qkvz_proj.weight.data.normal_(0.0, 0.02, generator=gen)

        class _DummyGDN(nn.Module):
            def __init__(self, gdn_cls, alt_stream):
                super().__init__()
                self.in_proj_qkvz = qkvz_proj
                self.in_proj_ba = ba_proj
                self._fused_in_proj_weight = None
                self._fused_in_proj_scale = None
                self._fused_input_proj_cpu_enabled = LazyValue(lambda: False)
                self.alt_stream = alt_stream
                self._forward_input_proj = types.MethodType(
                    gdn_cls._forward_input_proj, self
                )

        bucket_m = 32
        for gdn_cls in (
            qwen3_5_mod.Qwen3_5GatedDeltaNet,
            qwen3_next_mod.Qwen3GatedDeltaNet,
        ):
            eager_gdn = _DummyGDN(gdn_cls, alt_stream=None)
            _, eager_ba_13 = eager_gdn._forward_input_proj(
                x_full[:13].contiguous(), extend_batch
            )
            for use_alt_stream in (False, True):
                alt_stream = (
                    torch.cuda.Stream(device=device) if use_alt_stream else None
                )
                gdn = _DummyGDN(gdn_cls, alt_stream=alt_stream)
                static_x = torch.zeros(
                    (bucket_m, hidden_size), device=device, dtype=torch.bfloat16
                )
                capture_stream = torch.cuda.Stream(device=device)
                capture_stream.wait_stream(torch.cuda.current_stream(device))
                with model_capture_mode():
                    with torch.cuda.stream(capture_stream):
                        for _ in range(2):
                            gdn._forward_input_proj(static_x, extend_batch)
                    torch.cuda.current_stream(device).wait_stream(capture_stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        _, static_ba = gdn._forward_input_proj(static_x, extend_batch)

                static_x.zero_()
                static_x[:13].copy_(x_full[:13])
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertTrue(
                    torch.equal(static_ba[:13], eager_ba_13),
                    f"{gdn_cls.__name__} replay (alt_stream={use_alt_stream}) diverged",
                )


if __name__ == "__main__":
    unittest.main()
