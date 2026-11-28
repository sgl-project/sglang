import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.dp_attention import note_rank_barrier_in_prefill_graph
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    ModelOptFp8LinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
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
    configure_gdn_prefill_model,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_flags,
    override_platform,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


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

    def test_padding_below_first_bucket_still_honors_gdn_safety(self):
        runner = self._make_runner()
        batch = self._make_forward_batch(1)
        # Main permits 1 -> 4 despite the usual 2x waste bound. This is not
        # permission to replay a numerically unsafe GDN projection at M=4.
        self.assertTrue(runner.can_run_graph(batch))
        runner._requires_exact_prefill_bucket = True
        self.assertFalse(runner.can_run_graph(batch))

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

    def test_megamoe_idle_rank_replays_after_rank_barrier_capture(self):
        """An idle DP rank must not force every rank eager once the graphs hold
        rank-barrier breaks; those breaks capture only on steps every rank reaches."""
        runner = self._make_runner()
        flags, a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp()
        with (
            flags,
            a2a,
            dsa_cp,
            mla_cp,
            get_flags().dp.override(
                capturing_prefill_graph=True,
                prefill_graph_shape_has_rank_barrier=False,
            ),
        ):
            note_rank_barrier_in_prefill_graph()
            self.assertFalse(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 0])
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

    @contextmanager
    def _gdn_platform(
        self, *, is_cuda=True, is_sm100=True, is_sm120=False, sequence_parallel=False
    ):
        with (
            get_context().override_server_args(enable_deterministic_inference=False),
            mock.patch.object(runner_module, "is_cuda", return_value=is_cuda),
            override_platform(is_cuda=is_cuda, is_sm100=is_sm100, is_sm120=is_sm120),
            get_flags().sp.override(enabled=sequence_parallel),
            mock.patch(
                "sglang.kernels.ops.attention.fla.layernorm_gated._get_sm_count",
                return_value=188 if is_sm120 else 148,
            ),
            mock.patch(
                "sglang.kernels.ops.attention.fla.layernorm_gated.is_batch_invariant_mode_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.kernels.ops.attention.fla.layernorm_gated.check_cuda_graph_backend",
                return_value=False,
            ),
        ):
            yield

    def _make_projection(
        self,
        method_cls=ModelOptFp8LinearMethod,
        linear_cls=LinearBase,
    ):
        # Inspect real post-load module/method types without allocating GPU
        # weights or running distributed model initialization in CPU policy tests.
        projection = linear_cls.__new__(linear_cls)
        torch.nn.Module.__init__(projection)
        projection.quant_method = method_cls.__new__(method_cls)
        projection.quant_method.use_marlin = False
        projection.quant_method.use_sm120_fp8 = False
        projection.quant_method.quant_mode = "w4a4"
        projection.use_flashinfer_bmm = True
        projection.input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        projection.params_dtype = torch.bfloat16
        weight_dtype = {
            ModelOptFp8LinearMethod: torch.float8_e4m3fn,
            ModelOptFp4LinearMethod: torch.uint8,
        }.get(method_cls, torch.bfloat16)
        projection.weight = torch.nn.Parameter(
            torch.empty(8, 8, dtype=weight_dtype), requires_grad=False
        )
        projection.bias = None
        projection.gather_output = False
        return projection

    def _make_gdn_policy_runner(self, method_cls=ModelOptFp8LinearMethod, gdn_cls=None):
        from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        from sglang.srt.models.qwen3_next import Qwen3GatedDeltaNet

        if gdn_cls is None:
            from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

            gdn_cls = Qwen3_5GatedDeltaNet
        gdn = gdn_cls.__new__(gdn_cls)
        torch.nn.Module.__init__(gdn)
        gdn.in_proj_qkvz = self._make_projection(method_cls, MergedColumnParallelLinear)
        gdn.in_proj_ba = self._make_projection(
            UnquantizedLinearMethod, MergedColumnParallelLinear
        )
        gdn.out_proj = self._make_projection(method_cls, RowParallelLinear)
        # Both remain BF16 in otherwise quantized checkpoints: conv1d consumes
        # weights directly, and in_proj_ba uses the row-invariant prefill helper.
        gdn.conv1d = self._make_projection(
            UnquantizedLinearMethod, ColumnParallelLinear
        )
        gdn._fused_in_proj_weight = None
        gdn.num_v_heads, gdn.head_v_dim, gdn.attn_tp_size = 48, 128, 1
        norm_cls = FusedRMSNormGated if gdn_cls is Qwen3GatedDeltaNet else RMSNorm
        gdn.norm = norm_cls(128, device="meta", dtype=torch.bfloat16)
        layer = torch.nn.Module()
        layer.linear_attn = gdn
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = self._make_projection(method_cls)
        layer.self_attn.o_proj = self._make_projection(method_cls)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = self._make_projection(method_cls)
        layer.mlp.down_proj = self._make_projection(method_cls)
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([layer])
        backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        backend.linear_attn_backend = GDNAttnBackend.__new__(GDNAttnBackend)
        return SimpleNamespace(
            attn_backend=backend,
            model=model,
            model_config=SimpleNamespace(quantization="modelopt_mixed"),
        )

    def _assert_gdn_bucket_policy(self, model_runner, *, exact):
        self.assertEqual(_requires_exact_gdn_prefill_bucket(model_runner), exact)
        runner = self._make_runner()
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.capture_num_tokens = [16, 20]
        runner.max_num_tokens = 20
        runner._init_gdn_prefill_policy(model_runner)
        # Exercise admission of a real EXTEND shape, not just the stored flag.
        self.assertEqual(runner.can_run_graph(self._make_forward_batch(17)), not exact)
        for num_tokens in (16, 20):
            self.assertTrue(runner.can_run_graph(self._make_forward_batch(num_tokens)))

    def test_effective_fp8_methods_allow_padding_independent_of_checkpoint_label(self):
        model_runner = self._make_gdn_policy_runner()
        for quantization in (None, "modelopt_fp4", "modelopt_mixed", "fp8"):
            for mixed_chunk in (False, True):
                with (
                    self.subTest(quantization=quantization, mixed_chunk=mixed_chunk),
                    self._gdn_platform(),
                    get_context().override_server_args(enable_mixed_chunk=mixed_chunk),
                ):
                    model_runner.model_config.quantization = quantization
                    self._assert_gdn_bucket_policy(model_runner, exact=False)

    def test_bf16_in_each_remaining_projection_requires_exact_buckets(self):
        projection_paths = (
            "linear_attn.in_proj_qkvz",
            "linear_attn.out_proj",
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        )
        for path in projection_paths:
            for mixed_chunk in (False, True):
                with (
                    self.subTest(projection=path, mixed_chunk=mixed_chunk),
                    self._gdn_platform(),
                    get_context().override_server_args(enable_mixed_chunk=mixed_chunk),
                ):
                    model_runner = self._make_gdn_policy_runner()
                    # Keep the FP4 checkpoint label while changing only the
                    # effective method of one loaded projection.
                    model_runner.model_config.quantization = "modelopt_fp4"
                    projection = model_runner.model.model.layers[0].get_submodule(path)
                    projection.quant_method = UnquantizedLinearMethod()
                    projection.weight = torch.nn.Parameter(
                        torch.empty_like(projection.weight, dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                    self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_effective_fp4_w4a4_flashinfer_allows_padding(self):
        for backend in (
            Fp4GemmRunnerBackend.FLASHINFER_CUDNN,
            Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
            Fp4GemmRunnerBackend.FLASHINFER_TRTLLM,
            Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with (
                self.subTest(backend=backend),
                self._gdn_platform(),
                mock.patch.object(fp4_utils, "FP4_GEMM_RUNNER_BACKEND", backend),
            ):
                model_runner = self._make_gdn_policy_runner(ModelOptFp4LinearMethod)
                model_runner.model_config.quantization = None
                self._assert_gdn_bucket_policy(model_runner, exact=False)

    def test_fp8_kernel_fallback_requires_exact_buckets(self):
        for use_marlin, use_flashinfer_bmm in ((True, True), (False, False)):
            with (
                self.subTest(
                    use_marlin=use_marlin, use_flashinfer_bmm=use_flashinfer_bmm
                ),
                self._gdn_platform(),
            ):
                model_runner = self._make_gdn_policy_runner()
                projection = model_runner.model.model.layers[0].linear_attn.out_proj
                projection.quant_method.use_marlin = use_marlin
                projection.use_flashinfer_bmm = use_flashinfer_bmm
                self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_fp4_kernel_fallback_requires_exact_buckets(self):
        for backend, quant_mode in (
            (Fp4GemmRunnerBackend.AUTO, "w4a4"),
            (Fp4GemmRunnerBackend.MARLIN, "w4a4"),
            (Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL, "w4a16"),
        ):
            with (
                self.subTest(backend=backend, quant_mode=quant_mode),
                self._gdn_platform(),
                mock.patch.object(fp4_utils, "FP4_GEMM_RUNNER_BACKEND", backend),
            ):
                model_runner = self._make_gdn_policy_runner(ModelOptFp4LinearMethod)
                projection = model_runner.model.model.layers[0].linear_attn.out_proj
                projection.quant_method.quant_mode = quant_mode
                self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_gdn_padding_policy_is_scoped_to_cuda_and_sm100(self):
        for is_cuda, is_sm100, exact in (
            (False, False, False),
            (True, False, True),
            (True, True, False),
        ):
            with (
                self.subTest(is_cuda=is_cuda, is_sm100=is_sm100),
                self._gdn_platform(is_cuda=is_cuda, is_sm100=is_sm100),
            ):
                model_runner = self._make_gdn_policy_runner()
                self._assert_gdn_bucket_policy(model_runner, exact=exact)

    def test_non_gdn_attention_keeps_padding(self):
        hybrid_non_gdn = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        hybrid_non_gdn.linear_attn_backend = SimpleNamespace()
        for backend in (SimpleNamespace(), hybrid_non_gdn):
            with (
                self.subTest(backend=type(backend).__name__),
                self._gdn_platform(is_sm100=False),
            ):
                # Non-GDN models need neither a checkpoint label nor a loaded
                # GDN projection hierarchy to retain ordinary graph admission.
                self._assert_gdn_bucket_policy(
                    SimpleNamespace(attn_backend=backend), exact=False
                )

    def test_fp8_without_static_input_scale_or_with_bias_requires_exact_buckets(self):
        for fallback in ("dynamic_scale", "bias"):
            with self.subTest(fallback=fallback), self._gdn_platform():
                model_runner = self._make_gdn_policy_runner()
                projection = model_runner.model.model.layers[0].linear_attn.out_proj
                if fallback == "dynamic_scale":
                    projection.input_scale = None
                else:
                    projection.bias = torch.nn.Parameter(
                        torch.zeros(8, dtype=torch.bfloat16), requires_grad=False
                    )
                self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_quant_method_without_quantized_weights_requires_exact_buckets(self):
        for method_cls in (ModelOptFp8LinearMethod, ModelOptFp4LinearMethod):
            with (
                self.subTest(method=method_cls.__name__),
                self._gdn_platform(),
                mock.patch.object(
                    fp4_utils,
                    "FP4_GEMM_RUNNER_BACKEND",
                    Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
                ),
            ):
                model_runner = self._make_gdn_policy_runner(method_cls)
                projection = model_runner.model.model.layers[0].linear_attn.out_proj
                projection.weight = torch.nn.Parameter(
                    torch.empty(8, 8, dtype=torch.bfloat16), requires_grad=False
                )
                self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_ba_exemption_requires_the_actual_persistent_helper_path(self):
        class WrappedMergedLinear(MergedColumnParallelLinear):
            pass

        for fallback in (
            "wrapped_linear",
            "params_dtype",
            "weight_dtype",
            "bias",
            "gather_output",
            "sequence_parallel",
        ):
            with (
                self.subTest(fallback=fallback),
                self._gdn_platform(sequence_parallel=fallback == "sequence_parallel"),
            ):
                model_runner = self._make_gdn_policy_runner()
                gdn = model_runner.model.model.layers[0].linear_attn
                if fallback == "wrapped_linear":
                    gdn.in_proj_ba = self._make_projection(
                        UnquantizedLinearMethod, WrappedMergedLinear
                    )
                elif fallback == "params_dtype":
                    gdn.in_proj_ba.params_dtype = torch.float16
                elif fallback == "weight_dtype":
                    gdn.in_proj_ba.weight = torch.nn.Parameter(
                        torch.empty(8, 8, dtype=torch.float16), requires_grad=False
                    )
                elif fallback == "bias":
                    gdn.in_proj_ba.bias = torch.nn.Parameter(
                        torch.zeros(8, dtype=torch.bfloat16), requires_grad=False
                    )
                elif fallback == "gather_output":
                    gdn.in_proj_ba.gather_output = True
                self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_unknown_effective_projection_method_requires_exact_buckets(self):
        # Native FP8 has additional block/MX/Marlin/fallback dispatch branches;
        # its class alone does not certify the qualified ModelOpt BMM path.
        native_fp8 = Fp8LinearMethod.__new__(Fp8LinearMethod)
        for method in (None, object(), native_fp8):
            with self.subTest(method=type(method).__name__), self._gdn_platform():
                model_runner = self._make_gdn_policy_runner()
                projection = model_runner.model.model.layers[0].mlp.down_proj
                projection.quant_method = method
                self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_remaining_decoder_layers_are_inspected(self):
        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner()
            later_layer = self._make_gdn_policy_runner().model.model.layers[0]
            later_layer.mlp.down_proj = self._make_projection(UnquantizedLinearMethod)
            model_runner.model.model.layers.append(later_layer)
            self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_only_language_transformer_projections_affect_padding(self):
        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner()
            model_runner.model.lm_head = self._make_projection(UnquantizedLinearMethod)
            wrapper = torch.nn.Module()
            wrapper.language_model = model_runner.model
            wrapper.visual = self._make_projection(UnquantizedLinearMethod)
            model_runner.model = wrapper
            self._assert_gdn_bucket_policy(model_runner, exact=False)

    def test_gdn_backend_without_recognized_gdn_module_requires_exact_buckets(self):
        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner()
            layer = model_runner.model.model.layers[0]
            layer.linear_attn = torch.nn.Module()
            layer.linear_attn.in_proj_qkvz = self._make_projection()
            self._assert_gdn_bucket_policy(model_runner, exact=True)

    def test_qwen_next_and_direct_gdn_backend_allow_safe_padding(self):
        from sglang.srt.models.qwen3_next import Qwen3GatedDeltaNet

        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner(gdn_cls=Qwen3GatedDeltaNet)
            model_runner.attn_backend = GDNAttnBackend.__new__(GDNAttnBackend)
            self._assert_gdn_bucket_policy(model_runner, exact=False)

    def test_legacy_rmsnorm_only_rejects_a_row_tile_transition(self):
        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner()
            self.assertFalse(
                hasattr(model_runner.model.model.layers[0].linear_attn, "tp_size")
            )
            runner = self._make_runner()
            runner.capture_num_tokens = [8, 16, 20]
            runner.max_num_tokens = 20
            runner._init_gdn_prefill_policy(model_runner)
            # 148 SMs and 48 value heads: 6 -> 8 changes one row/CTA to two;
            # 7 -> 8 and 13 -> 16 keep their norm reduction layouts.
            for tokens, expected in ((6, False), (7, True), (8, True), (13, True)):
                with self.subTest(tokens=tokens):
                    self.assertEqual(
                        runner.can_run_graph(self._make_forward_batch(tokens)), expected
                    )
            with mock.patch(
                "sglang.kernels.ops.attention.fla.layernorm_gated.is_batch_invariant_mode_enabled",
                return_value=True,
            ):
                self.assertTrue(runner.can_run_graph(self._make_forward_batch(6)))

    def test_dp_gathered_counts_checked_against_common_replay_bucket(self):
        with (
            self._gdn_platform(),
            mock.patch.object(
                forward_batch_info,
                "prefill_graph_tolerates_sum_len",
                return_value=False,
            ),
            mock.patch.object(
                runner_module,
                "prefill_graph_tolerates_sum_len",
                return_value=False,
            ),
        ):
            runner = self._make_runner()
            runner.prefill_backend_name = Backend.BREAKABLE
            runner.capture_num_tokens = [4, 8]
            runner.max_num_tokens = 8
            runner._init_gdn_prefill_policy(self._make_gdn_policy_runner())
            # Rank 0 (3 tokens -> local bucket 4) and Rank 1 (8 tokens) both
            # pass local checks, but replaying the shared 8-token bucket crosses
            # the 1 -> 2 row/CTA norm boundary on 148 SMs with 48 local v_heads.
            self.assertTrue(runner.can_run_graph(self._make_forward_batch(3)))
            self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))
            self.assertFalse(runner.can_replay_dp_tokens([3, 8]))
            self.assertTrue(runner.can_replay_dp_tokens([3, 4]))
            self.assertTrue(runner.can_replay_dp_tokens([7, 8]))

            dp_batch = self._make_forward_batch(3)
            dp_batch.global_num_tokens_cpu = [3, 8]
            dp_batch.can_run_dp_prefill_cuda_graph = True
            self.assertFalse(runner.can_run_graph(dp_batch))

            runner._requires_exact_prefill_bucket = True
            self.assertFalse(runner.can_replay_dp_tokens([4, 8]))
            self.assertTrue(runner.can_replay_dp_tokens([8, 8]))

    def test_shared_gdn_subclass_and_rocm_wrapper_hooks(self):
        import sglang.srt.models.qwen3_5 as qwen3_5_mod
        from sglang.srt.model_loader.loader import DefaultModelLoader
        from sglang.srt.models.gigachat35 import GigaChat35GatedDeltaNet

        # Dense Qwen3.5 VLM wrapper must not forward prepare_before_cuda_graph_capture
        # into Qwen3_5ForCausalLM's ROCm fused-projection packer.
        self.assertNotIn(
            "prepare_before_cuda_graph_capture",
            qwen3_5_mod.Qwen3_5ForConditionalGeneration.__dict__,
        )

        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner(
                method_cls=UnquantizedLinearMethod,
                gdn_cls=GigaChat35GatedDeltaNet,
            )
            DefaultModelLoader.postprocess_weights(
                model_runner.model, torch.device("cpu")
            )
            gdn = model_runner.model.model.layers[0].linear_attn
            self.assertEqual(gdn._gdn_prefill_max_tokens, 0)
            self.assertEqual(gdn.in_proj_ba._gdn_prefill_max_tokens, 0)

    def test_later_gdn_norm_and_unknown_norm_cannot_bypass_admission(self):
        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner()
            later = self._make_gdn_policy_runner().model.model.layers[0]
            later.linear_attn.num_v_heads = 40
            model_runner.model.model.layers.append(later)
            runner = self._make_runner()
            runner.capture_num_tokens = [8, 16]
            runner._init_gdn_prefill_policy(model_runner)
            self.assertFalse(runner.can_run_graph(self._make_forward_batch(7)))
            self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))
            later.linear_attn.norm = torch.nn.Identity()
            runner._init_gdn_prefill_policy(model_runner)
            self.assertFalse(runner.can_run_graph(self._make_forward_batch(13)))
            self.assertTrue(runner.can_run_graph(self._make_forward_batch(16)))

    def test_sm103_fused_norm_checks_both_small_batch_heuristics(self):
        from sglang.srt.models.qwen3_next import Qwen3GatedDeltaNet

        with (
            self._gdn_platform(),
            mock.patch(
                "sglang.kernels.ops.attention.fla.fused_norm_gate.is_arch_support_pdl",
                return_value=True,
            ),
            mock.patch(
                "sglang.kernels.ops.attention.fla.fused_norm_gate.current_arch",
                return_value=SimpleNamespace(
                    is_sm=lambda major, minor: (major, minor) == (10, 3)
                ),
            ),
        ):
            for activation, buckets, cases in (
                ("swish", [2, 4, 8], ((1, False), (3, True))),
                ("sigmoid", [8, 16], ((4, False), (5, True))),
            ):
                model_runner = self._make_gdn_policy_runner(gdn_cls=Qwen3GatedDeltaNet)
                gdn = model_runner.model.model.layers[0].linear_attn
                gdn.num_v_heads = 64
                gdn.norm.activation = activation
                runner = self._make_runner()
                runner.capture_num_tokens = buckets
                runner.max_num_tokens = max(buckets)
                runner._init_gdn_prefill_policy(model_runner)
                for tokens, expected in cases:
                    with self.subTest(activation=activation, tokens=tokens):
                        self.assertEqual(
                            runner.can_run_graph(self._make_forward_batch(tokens)),
                            expected,
                        )

    def test_existing_sm120_deterministic_method_can_keep_padding(self):
        for device_capability in ((12, 0), (12, 1)):
            with (
                self.subTest(device_capability=device_capability),
                self._gdn_platform(is_sm100=False, is_sm120=True),
                override_platform(device_capability=device_capability),
            ):
                model_runner = self._make_gdn_policy_runner()
                for projection in model_runner.model.model.modules():
                    if isinstance(projection, LinearBase) and (
                        type(projection.quant_method) is ModelOptFp8LinearMethod
                    ):
                        projection.quant_method.use_sm120_fp8 = True
                        projection.use_flashinfer_bmm = False
                self._assert_gdn_bucket_policy(model_runner, exact=True)
                self.assertEqual(
                    configure_gdn_prefill_model(
                        model_runner.model, attn_backend=model_runner.attn_backend
                    ),
                    0,
                )
                self.assertEqual(
                    model_runner.model.model.layers[
                        0
                    ].linear_attn.in_proj_ba._gdn_prefill_max_tokens,
                    0,
                )
                with get_exec().deterministic.override(
                    enable_deterministic_inference=True
                ):
                    self._assert_gdn_bucket_policy(model_runner, exact=False)
                    self.assertIsNone(
                        configure_gdn_prefill_model(
                            model_runner.model, attn_backend=model_runner.attn_backend
                        )
                    )
                    self.assertIsNone(
                        model_runner.model.model.layers[
                            0
                        ].linear_attn.in_proj_ba._gdn_prefill_max_tokens
                    )

    def test_external_prefill_body_is_not_certified_by_python_projection_types(self):
        with self._gdn_platform():
            model_runner = self._make_gdn_policy_runner()
            self.assertIsNone(
                configure_gdn_prefill_model(
                    model_runner.model, attn_backend=model_runner.attn_backend
                )
            )
            runner_module.set_external_prefill_body(model_runner.model, object())
            self._assert_gdn_bucket_policy(model_runner, exact=True)
            self.assertEqual(
                model_runner.model.model.layers[
                    0
                ].linear_attn.in_proj_ba._gdn_prefill_max_tokens,
                0,
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

            # Exact-bucket models (such as SM121/GB10 or BF16 checkpoints) keep
            # native linear ba_proj on EXTEND with zero RuntimeContext lookups;
            # deterministic mode stamps None at model-configuration time.
            ba_proj._gdn_prefill_max_tokens = 0
            try:
                with (
                    mock.patch(
                        "sglang.srt.runtime_context.get_context",
                        side_effect=AssertionError("unexpected per-layer get_context"),
                    ),
                    mock.patch.object(
                        ba_proj, "forward", wraps=ba_proj.forward
                    ) as spy_exact,
                ):
                    out_exact = helper(ba_proj, x_full[:13].contiguous(), extend_batch)
                    spy_exact.assert_called_once()
                expected_exact, _ = ba_proj(x_full[:13].contiguous())
                self.assertTrue(torch.equal(out_exact, expected_exact))
            finally:
                del ba_proj._gdn_prefill_max_tokens

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


class TestPrefillGraphRankBarrierBreaks(CustomTestCase):
    """Breaks that capture behind TP-group barriers live only in shapes at or
    above a token threshold; under MegaMoE each DP rank replays its own bucket."""

    _SHAPES = [512, 1024, 2048, 4096, 8192, 16384]
    _BARRIER_SHAPES = frozenset({2048, 4096, 8192, 16384})

    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.capture_num_tokens = self._SHAPES
        runner.enable_cp_bcg_capture = False
        runner._capture_chunked_prefix = False
        runner._rank_barrier_shapes = frozenset()
        return runner

    def test_step_lets_breaks_capture_only_when_every_rank_reaches_them(self):
        """A rank whose bucket is below the threshold keeps those breaks inline and
        never reaches their barriers; letting the others capture hung the step."""
        runner = self._make_runner()
        runner._rank_barrier_shapes = self._BARRIER_SHAPES
        for counts, expected in (
            ([8000, 50, 3000, 16384], False),
            ([8000, 2000, 3000, 16384], True),
            ([8000, 0, 3000, 4096], False),
            ([8000, 1024, 3000, 4096], False),
        ):
            with self.subTest(counts=counts):
                batch = SimpleNamespace(global_num_tokens_cpu=counts, input_ids=[])
                self.assertIs(runner._every_rank_reaches_rank_barrier(batch), expected)
        runner._rank_barrier_shapes = frozenset()
        batch = SimpleNamespace(global_num_tokens_cpu=[8000] * 4, input_ids=[])
        self.assertFalse(runner._every_rank_reaches_rank_barrier(batch))

    def test_capture_notes_rank_barriers_per_shape(self):
        """Shapes capture largest first, so a barrier noted by a large shape must
        not mark the smaller shapes captured after it."""
        runner = self._make_runner()

        def capture_one_shape(num_tokens, prefix_num_chunks=0):
            if num_tokens >= 2048:
                note_rank_barrier_in_prefill_graph()

        runner.capture_one_shape = capture_one_shape
        with get_flags().dp.override(
            capturing_prefill_graph=True,
            prefill_graph_shape_has_rank_barrier=False,
        ):
            for num_tokens in reversed(self._SHAPES):
                runner._capture_shape(num_tokens)
        self.assertEqual(runner._rank_barrier_shapes, self._BARRIER_SHAPES)

    def test_startup_prime_runs_with_every_rank_reaching_the_breaks(self):
        """The prime is the startup step on which every rank's breaks capture
        together; without the flag they would skip capture and leave it to serving."""
        runner = self._make_runner()
        runner.use_captured_attn_metadata = True
        runner.device_module = mock.Mock()
        runner.backend = SimpleNamespace(replay_session=nullcontext)
        runner.model_runner = SimpleNamespace(model=SimpleNamespace())
        forward_batch = SimpleNamespace(
            lora_ids=None, moe_num_token_non_padded=lambda: None
        )
        runner.capture_prepare = mock.Mock(return_value=(forward_batch, mock.Mock()))
        seen = []
        runner._run_forward = mock.Mock(
            side_effect=lambda batch, num_tokens: seen.append(
                get_flags().dp.every_rank_reaches_rank_barrier
            )
        )
        parallel = SimpleNamespace(tp_group=mock.Mock(), tp_rank=1)
        with mock.patch.object(runner_module, "get_parallel", return_value=parallel):
            runner._prime_rank_barrier_breaks()
            runner._run_forward.assert_not_called()
            runner._rank_barrier_shapes = self._BARRIER_SHAPES
            runner._prime_rank_barrier_breaks()
        self.assertEqual(seen, [True])
        self.assertIn(runner.capture_prepare.call_args.args[0], self._BARRIER_SHAPES)
        self.assertFalse(get_flags().dp.every_rank_reaches_rank_barrier)


if __name__ == "__main__":
    unittest.main()
