"""Route reuse, capability guards and graph-compile metadata contracts of the SAIL
fused API (CPU)."""

import inspect
import sys
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.gemm import sail_deepgemm as ops
from sglang.srt.layers.deep_gemm_wrapper import compile_utils, entrypoint

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSailDeepGemmConfigAbi(CustomTestCase):
    def test_non_fused_configs_survive_vendor_optional_argument_insertion(self):
        """Tuning configs must not be taken as new positional arguments after the
        vendor inserts parameters before configs."""
        cases = (
            (
                "grouped_gemm_nt_f8f8bf16_nopad",
                "m_grouped_gemm_fp8_fp8_bf16_nt_nopad",
                "nopad",
                True,
                False,
            ),
            ("gemm_nt_i8i8bf16", "gemm_int8_int8_bf16_nt", "dense", True, False),
            (
                "grouped_gemm_nt_i8i8bf16_contig",
                "m_grouped_gemm_int8_int8_bf16_nt_contiguous",
                "contig",
                True,
                False,
            ),
            (
                "grouped_gemm_nt_i8i8bf16_masked",
                "m_grouped_gemm_int8_int8_bf16_nt_masked",
                "masked",
                True,
                False,
            ),
            (
                "grouped_gemm_nt_i8i8bf16_nopad",
                "m_grouped_gemm_int8_int8_bf16_nt_nopad",
                "nopad",
                True,
                False,
            ),
            (
                "grouped_gemm_nt_bf16_nopad",
                "m_grouped_gemm_bf16_bf16_bf16_nt_nopad",
                "nopad",
                False,
                False,
            ),
            ("gemm_nt_f4f4bf16", "gemm_fp4_fp4_bf16_nt", "dense", True, True),
            (
                "grouped_gemm_nt_f4f4bf16_masked",
                "m_grouped_gemm_fp4_fp4_bf16_nt_masked",
                "masked",
                True,
                True,
            ),
            (
                "grouped_gemm_nt_f4f4bf16_nopad",
                "m_grouped_gemm_fp4_fp4_bf16_nt_nopad",
                "nopad",
                True,
                True,
            ),
        )
        for wrapper, vendor_name, layout, paired, has_bias in cases:
            for explicit in (False, True):
                with self.subTest(wrapper=wrapper, explicit=explicit):
                    a = torch.ones((2, 3, 2) if layout == "masked" else (3, 2))
                    b = torch.ones((2, 2) if layout == "dense" else (2, 2, 2))
                    lhs = (a, torch.ones_like(a)) if paired else a
                    rhs = (b, torch.ones_like(b)) if paired else b
                    out = torch.zeros_like(a)
                    ids = torch.tensor([0, 1, 0], dtype=torch.int32)
                    rows = torch.tensor([2, 1], dtype=torch.int32)
                    bias = torch.zeros(2)
                    args = [lhs, rhs]
                    if has_bias:
                        args.append(bias)
                    args.append(out)
                    if layout == "masked":
                        args.extend((rows, 3))
                    elif layout == "nopad":
                        args.extend((ids, rows))
                    elif layout == "contig":
                        args.append(ids)
                    config = [16, 128, 128]

                    def vendor(*received, configs=None, **kwargs):
                        # Strictly check the original positional
                        # argument count, simulating a new vendor
                        # parameter that defaults to None.
                        self.assertEqual(len(received), len(args))
                        self.assertIs(configs, config)
                        for actual, expected in zip(received, args):
                            self.assertIs(actual, expected)
                        if layout == "masked":
                            self.assertTrue(kwargs["enable_sbo_overlap"])
                            self.assertIs(kwargs["signal"], signal)
                            self.assertEqual(kwargs["max_block_n"], 128)
                        else:
                            self.assertEqual(kwargs, {})
                        # Write the output that really crosses the
                        # external boundary, verifying the wrapper did
                        # not lose the output object.
                        if layout in ("dense", "masked"):
                            out.copy_(a @ b.transpose(-1, -2))
                        else:
                            out.copy_(
                                torch.stack([a[i] @ b[ids[i]].T for i in range(3)])
                            )

                    signal = torch.zeros(1, dtype=torch.int32)
                    vendor_module = SimpleNamespace(**{vendor_name: vendor})
                    with (
                        patch.object(entrypoint, "_is_sail", True),
                        patch.object(
                            entrypoint, "deep_gemm", vendor_module, create=True
                        ),
                        patch.object(
                            entrypoint,
                            "tuner",
                            SimpleNamespace(
                                get_deep_gemm_config=lambda *a, **kw: config
                            ),
                            create=True,
                        ),
                        patch.object(entrypoint, "_SANITY_CHECK", False),
                        patch.object(
                            compile_utils,
                            "deep_gemm_execution_hook",
                            return_value=nullcontext(),
                        ),
                        patch.object(
                            entrypoint,
                            "configure_deep_gemm_num_sms",
                            return_value=nullcontext(),
                        ),
                    ):
                        kwargs = {"configs": config} if explicit else {}
                        if layout == "masked":
                            kwargs.update(
                                overlap_args=SimpleNamespace(num_sms=8, signal=signal),
                                max_block_n=128,
                            )
                        getattr(entrypoint, wrapper)(*args, **kwargs)
                    torch.testing.assert_close(out, torch.full_like(out, 2))


class TestDeepGemmMaskedActivationContract(CustomTestCase):
    def test_oai_validation_and_unsupported_mxfp4_fail_before_launch(self):
        """Invalid activation combinations must raise instead of degrading to plain
        SiLU or writing a wrong FP4 layout."""
        from sglang.kernels.ops.moe import ep_moe_kernels as kernels
        from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        for limit, swiglu, swizzle, message in (
            (None, None, False, "gemm1_clamp_limit"),
            (7.0, 5.0, False, "mutually exclusive"),
            (7.0, None, True, "swizzle"),
        ):
            with (
                self.subTest(message=message),
                patch.object(
                    runner_module,
                    "get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_megamoe=lambda: swizzle),
                ),
                patch.object(
                    runner_module.deep_gemm_sm120, "use_swizzle", return_value=True
                ),
                self.assertRaisesRegex(AssertionError, message),
            ):
                runner_module.DeepGemmRunnerCore(
                    MoeRunnerConfig(
                        gemm1_alpha=1.702,
                        gemm1_clamp_limit=limit,
                        swiglu_limit=swiglu,
                    )
                )
        with patch.object(
            runner_module,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_megamoe=lambda: False),
        ):
            runner_module.DeepGemmRunnerCore(
                MoeRunnerConfig(
                    activation="situ",
                    gemm1_alpha=4.0,
                    gemm1_clamp_limit=None,
                )
            )
            runner = runner_module.DeepGemmRunnerCore(
                MoeRunnerConfig(
                    gemm1_alpha=1.702,
                    gemm1_clamp_limit=7.0,
                )
            )
        # A None payload makes any early entry into GEMM/allocator fail at once.
        with self.assertRaisesRegex(AssertionError, "masked mxfp4"):
            runner._run_masked_fp4_gemm(None, None, {})
        with (
            patch.object(kernels, "_is_sail", True),
            patch.object(kernels, "get_device_sm", return_value=89),
            self.assertRaisesRegex(AssertionError, "no MXFP4 e2m1 packing branch"),
        ):
            kernels.silu_and_mul_masked_post_quant_fwd(
                torch.ones((2, 3, 256), dtype=torch.bfloat16),
                torch.empty((2, 3, 64), dtype=torch.uint8),
                torch.empty((2, 2, 3), dtype=torch.uint16),
                32,
                torch.tensor([1, 2], dtype=torch.int32),
                use_fp8=False,
                use_mxfp4=True,
                gemm1_alpha=1.702,
                gemm1_clamp_limit=7.0,
            )

    def test_contiguous_oai_output_and_layout_reach_all_gemm2_paths(self):
        """All six GEMM2 kinds must receive the oai output for their weight layout,
        and clamp must not modify the GEMM1 buffer."""
        from sglang.kernels.ops.quantization import fp8_kernel, int8_kernel, sail_mxfp4
        from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        gate = torch.tensor([-10, -1, 0.5, 10], dtype=torch.bfloat16).repeat(64)
        up = torch.tensor([-10, 2, -0.5, 10], dtype=torch.bfloat16).repeat(64)
        clipped_gate = gate.float().clamp(max=7)
        expected = (
            clipped_gate
            * torch.sigmoid(clipped_gate * 1.702)
            * (up.float().clamp(-7, 7) + 1)
        ).to(torch.bfloat16)
        variants = (
            ("bf16", torch.bfloat16, "grouped_gemm_nt_bf16_nopad"),
            ("fp8", torch.float8_e4m3fn, "grouped_gemm_nt_f8f8bf16_nopad"),
            ("channel", torch.float8_e4m3fn, "grouped_gemm_nt_i8i8bf16_nopad"),
            ("int8", torch.int8, "grouped_gemm_nt_i8i8bf16_nopad"),
            ("fp4", torch.uint8, "grouped_gemm_nt_f4f4bf16_nopad"),
            ("w4a16", torch.uint8, "grouped_gemm_nt_bf16i4bf16_nopad"),
        )
        for variant, dtype, gemm_name in variants:
            for interleaved in (False, True):
                packed = (
                    torch.stack((gate, up), -1).flatten()
                    if interleaved
                    else torch.cat((gate, up))
                )
                produced = []
                scales = torch.empty((4, 3), dtype=torch.uint16).t()
                quant = runner_module.DeepGemmMoeQuantInfo(
                    w13_weight=torch.empty((2, 512, 32), dtype=dtype),
                    w2_weight=torch.empty((2, 64, 128), dtype=dtype),
                    w13_scale=torch.ones((2, 512, 1)),
                    w2_scale=torch.ones((2, 64, 1)),
                    use_fp8=variant in ("fp8", "channel"),
                    per_channel_quant=variant == "channel",
                    use_int8=variant == "int8",
                    use_mxfp4=variant == "fp4",
                    use_mxfp4_w4a16=variant == "w4a16",
                )
                runner = object.__new__(runner_module.DeepGemmRunnerCore)
                runner.config = MoeRunnerConfig(
                    gemm1_alpha=1.702,
                    gemm1_clamp_limit=7.0,
                    gate_up_interleaved=interleaved,
                )
                runner.swiglu_limit = None
                runner.use_swizzle = False

                def gemm(lhs, rhs, *args, **kw):
                    weight = rhs[0] if isinstance(rhs, tuple) else rhs
                    out = args[1] if variant == "fp4" else args[0]
                    if weight is quant.w13_weight:
                        out.copy_(packed.expand(3, -1))
                        produced.append(out)
                    else:
                        activation = lhs[0] if isinstance(lhs, tuple) else lhs
                        torch.testing.assert_close(activation, expected.expand(3, -1))
                        torch.testing.assert_close(produced[0], packed.expand(3, -1))
                        if variant == "fp4":
                            self.assertIs(lhs[1], scales)
                            self.assertEqual(lhs[1].stride(), (1, 3))
                        out.copy_(activation[:, : out.shape[-1]])

                def quantize(x, *args, **kw):
                    # The device quant boundary only keeps the input; the
                    # real activation and runner branches are not mocked.
                    return x, torch.ones((3, 2))

                def fp4_quantize(x, axis=-1, round_up=True, enable_fused_moe=False):
                    self.assertEqual((axis, enable_fused_moe), (1, False))
                    return x, scales

                with ExitStack() as stack:
                    stack.enter_context(
                        self.subTest(variant=variant, interleaved=interleaved)
                    )
                    for name, value in (
                        ("_is_sail", True),
                        ("dispose_tensor", lambda x: None),
                        ("get_parallel", lambda: SimpleNamespace(tp_group=None)),
                        ("use_symmetric_memory", lambda *a, **kw: nullcontext()),
                        ("is_allocation_symmetric", lambda: False),
                    ):
                        stack.enter_context(patch.object(runner_module, name, value))
                    stack.enter_context(
                        patch.object(runner_module.deep_gemm_wrapper, gemm_name, gemm)
                    )
                    for flag in (
                        "DEEPGEMM_SCALE_UE8M0",
                        "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                    ):
                        stack.enter_context(
                            patch.object(runner_module.deep_gemm_wrapper, flag, False)
                        )
                    stack.enter_context(
                        patch.object(
                            fp8_kernel, "sglang_per_token_group_quant_fp8", quantize
                        )
                    )
                    stack.enter_context(
                        patch.object(fp8_kernel, "sglang_per_token_quant_fp8", quantize)
                    )
                    stack.enter_context(
                        patch.object(int8_kernel, "per_token_quant_int8", quantize)
                    )
                    stack.enter_context(
                        patch.object(sail_mxfp4, "downcast_to_mxfp4", fp4_quantize)
                    )
                    stack.enter_context(
                        patch.dict(
                            sys.modules,
                            {
                                "deep_gemm": SimpleNamespace(
                                    preprocess_mxfp4_scales=lambda x: x
                                ),
                            },
                        )
                    )
                    payload = runner_module.DeepGemmRunnerInput(
                        torch.empty(
                            (3, 32),
                            dtype=torch.bfloat16 if variant == "w4a16" else dtype,
                        ),
                        torch.ones((3, 1)),
                        False,
                        m_indices=torch.zeros(3, dtype=torch.int32),
                    )
                    result = runner.run(
                        payload,
                        quant,
                        {
                            "all_tokens": 3,
                            "hidden_states_shape": (3, 32),
                            "hidden_states_device": torch.device("cpu"),
                            "hidden_states_dtype": dtype,
                        },
                    ).hidden_states
                torch.testing.assert_close(
                    result, expected[: result.shape[-1]].expand(3, -1)
                )
        with self.assertRaisesRegex(AssertionError, "expected bf16"):
            runner_module._apply_oai_swiglu(packed.float(), 1.702, 7.0, False)

    def test_masked_runner_forwards_alpha_and_keeps_legacy_launch_defaults(self):
        """BF16, channel FP8/INT8 and W4A16 must propagate the activation protocol
        down to the actual launch."""
        from sglang.kernels.ops.moe import ep_moe_kernels as kernels
        from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        for variant, dtype, gemm_name in (
            ("bf16", torch.bfloat16, "grouped_gemm_nt_bf16_masked"),
            ("channel", torch.float8_e4m3fn, "grouped_gemm_nt_f8f8bf16_masked"),
            ("int8", torch.int8, "grouped_gemm_nt_i8i8bf16_masked"),
            ("w4a16", torch.uint8, "grouped_gemm_nt_bf16i4bf16_masked"),
        ):
            for alpha in (None, 1.702):
                launches = []
                quant = runner_module.DeepGemmMoeQuantInfo(
                    w13_weight=torch.empty((2, 512, 32), dtype=dtype),
                    w2_weight=torch.empty((2, 64, 128), dtype=dtype),
                    w13_scale=torch.ones((2, 512, 1)),
                    w2_scale=torch.ones((2, 64, 1)),
                    use_fp8=variant == "channel",
                    per_channel_quant=variant == "channel",
                    use_int8=variant == "int8",
                    use_mxfp4_w4a16=variant == "w4a16",
                )
                runner = object.__new__(runner_module.DeepGemmRunnerCore)
                runner.config = MoeRunnerConfig(
                    gemm1_alpha=alpha, gemm1_clamp_limit=7.0
                )
                runner.swiglu_limit = None
                counts = torch.tensor([1, 2], dtype=torch.int32)

                class LaunchBoundary:
                    def __getitem__(self, grid):
                        def launch(*args, **kw):
                            launches.append((grid, kw))
                            args[4].fill_(2)

                        return launch

                def gemm(lhs, rhs, out, *args, **kw):
                    weight = rhs[0] if isinstance(rhs, tuple) else rhs
                    if weight is quant.w13_weight:
                        out.fill_(1)
                    else:
                        activation = lhs[0] if isinstance(lhs, tuple) else lhs
                        out.copy_(activation[..., : out.shape[-1]])
                    return (16, 1)

                with ExitStack() as stack:
                    stack.enter_context(self.subTest(variant=variant, alpha=alpha))
                    for name, value in (
                        ("_is_sail", True),
                        ("dispose_tensor", lambda x: None),
                        ("get_parallel", lambda: SimpleNamespace(tp_group=None)),
                        ("use_symmetric_memory", lambda *a, **kw: nullcontext()),
                        ("is_allocation_symmetric", lambda: False),
                    ):
                        stack.enter_context(patch.object(runner_module, name, value))
                    stack.enter_context(
                        patch.object(runner_module.deep_gemm_wrapper, gemm_name, gemm)
                    )
                    stack.enter_context(
                        patch.object(
                            runner_module.deep_gemm_wrapper,
                            "DEEPGEMM_SCALE_UE8M0",
                            False,
                        )
                    )
                    stack.enter_context(patch.object(kernels, "_is_sail", True))
                    kernel = (
                        "_silu_and_mul_post_quant_kernel"
                        if variant in ("channel", "int8")
                        else "_silu_and_mul_kernel"
                    )
                    stack.enter_context(patch.object(kernels, kernel, LaunchBoundary()))
                    payload = runner_module.DeepGemmRunnerInput(
                        torch.empty(
                            (2, 3, 64),
                            dtype=torch.bfloat16 if variant == "w4a16" else dtype,
                        ),
                        torch.ones((2, 3, 1)),
                        True,
                        masked_m=counts,
                        expected_m=1,
                    )
                    state = {
                        "hidden_states_device": torch.device("cpu"),
                        "meta_overlap_args": {},
                    }
                    result = runner.run(payload, quant, state).hidden_states
                torch.testing.assert_close(result, torch.full_like(result, 2))
                self.assertEqual(
                    state["meta_overlap_args"], {"block_m": 16, "threshold": 1}
                )
                self.assertEqual(len(launches), 1)
                grid, kw = launches[0]
                self.assertEqual(
                    grid, (1 if variant in ("channel", "int8") else 2, 64, 2)
                )
                self.assertEqual(kw["GEMM1_ALPHA"], alpha or 0.0)
                self.assertEqual(kw["GEMM1_CLAMP_LIMIT"], 7.0)

    def test_launch_scale_capacity_and_non_power_of_two_channel_width(self):
        """The per-token scale write bound must not exceed the capacity; a 1536-wide
        channel must have exactly one group."""
        from sglang.kernels.ops.moe import ep_moe_kernels as kernels

        # BLOCK_N, the hidden blocks and the per-block scale count are fixed by
        # the kernel write protocol.
        cases = (
            (1536, 1536, False, 2048, 1, 1),
            (1536, 1536, True, 2048, 1, 1),
            (2048, 2048, False, 2048, 1, 1),
            (1536, 128, False, 512, 3, 4),
            (1408, 128, False, 128, 11, 1),
        )
        for width, group, int8, block_n, blocks, groups_per_block in cases:
            launches = []

            class LaunchBoundary:
                def __getitem__(self, grid):
                    def launch(*args, **kwargs):
                        launches.append((grid, kwargs))

                    return launch

            x = torch.ones((2, 3, width * 2), dtype=torch.bfloat16)
            out = torch.empty(
                (2, 3, width), dtype=torch.int8 if int8 else torch.float8_e4m3fn
            )
            counts = torch.tensor([1, 3], dtype=torch.int32)
            slots = blocks * groups_per_block
            with (
                self.subTest(width=width, group=group, int8=int8),
                patch.object(kernels, "_is_sail", True),
                patch.object(
                    kernels, "_silu_and_mul_post_quant_kernel", LaunchBoundary()
                ),
            ):
                for capacity in (slots, slots - 1):
                    launches.clear()
                    scales = torch.empty((2, 3, capacity), dtype=torch.float32)
                    kwargs = dict(
                        use_int8=int8,
                        use_fp8=not int8,
                        gemm1_alpha=1.702,
                        gemm1_clamp_limit=7.0,
                    )
                    if capacity < slots:
                        with self.assertRaisesRegex(
                            AssertionError, "output_scale too small"
                        ):
                            kernels.silu_and_mul_masked_post_quant_fwd(
                                x, out, scales, group, counts, **kwargs
                            )
                        self.assertEqual(launches, [])
                        continue
                    kernels.silu_and_mul_masked_post_quant_fwd(
                        x, out, scales, group, counts, **kwargs
                    )
                    self.assertEqual(len(launches), 1)
                    grid, launch = launches[0]
                    self.assertEqual(grid, (blocks, 64, 2))
                    self.assertEqual(launch["BLOCK_N"], block_n)
                    self.assertEqual(
                        block_n // launch["QUANT_GROUP_SIZE"], groups_per_block
                    )
                    self.assertEqual(launch["GEMM1_ALPHA"], 1.702)
                    # The alpha gate/up FP32 double buffer must not exceed
                    # 256 KiB of shared memory.
                    self.assertLessEqual(
                        block_n * 4 * 2 * (launch["NUM_STAGE"] - 1), 256 * 1024
                    )


class TestSailDeepGemmFusedApi(CustomTestCase):
    def test_capability_failure_precedes_vendor_execution(self):
        """gather-read must not start on NVIDIA, with JIT disabled, or on an old
        vendor API."""
        gemm_name = "m_grouped_gemm_bf16_bf16_bf16_nt_fused"
        cases = (
            (False, True, (), ValueError, "SAIL"),
            (True, False, (), RuntimeError, "JIT"),
            (True, True, (gemm_name,), RuntimeError, "moe_align_block_size"),
            (True, True, ("moe_align_block_size",), RuntimeError, gemm_name),
        )
        a = torch.empty((1, 128), dtype=torch.bfloat16)
        b = torch.empty((2, 128, 128), dtype=torch.bfloat16)
        ids = torch.zeros((1, 1), dtype=torch.int32)
        for sail, jit, names, error, message in cases:
            with self.subTest(sail=sail, jit=jit, names=names):
                vendor = SimpleNamespace(
                    **{name: lambda *args, **kw: None for name in names}
                )
                with (
                    patch.object(entrypoint, "_is_sail", sail),
                    patch.object(entrypoint, "ENABLE_JIT_DEEPGEMM", jit),
                    patch.object(entrypoint, "deep_gemm", vendor, create=True),
                    self.assertRaisesRegex(error, message),
                ):
                    entrypoint.moe_align_block_size(a, b, ids)

    def test_bf16_routes_reuse_single_alignment_result(self):
        """The GEMM write result and the inverse permutation must come from the same
        routing; metadata must not be generated twice."""
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
        b = torch.stack([torch.eye(2) * i for i in (1, 2, 3)]).to(torch.bfloat16)
        ids = torch.tensor([[1, 0], [2, 1]], dtype=torch.int32)
        tokens = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
        experts = ids.flatten()[tokens.long()]
        rows = torch.tensor([1, 2, 1], dtype=torch.int32)
        inverse = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
        config = [16, 128, 128]
        blocks = torch.tensor([4], dtype=torch.int32)
        once = iter([(config, rows, experts, tokens, blocks, inverse, experts)])

        def align(*args, **kwargs):
            return next(once)

        def gemm(lhs, rhs, out, counts, expert_data, token_data, block_data, configs):
            self.assertIs(counts, rows)
            self.assertIs(expert_data, experts)
            self.assertIs(token_data, tokens)
            self.assertIs(block_data, blocks)
            self.assertIs(configs, config)
            out.copy_(
                torch.stack(
                    [
                        lhs[int(token) // 2].float() @ rhs[int(expert)].float().T
                        for token, expert in zip(token_data, expert_data)
                    ]
                )
            )

        vendor = SimpleNamespace(
            moe_align_block_size=align, m_grouped_gemm_bf16_bf16_bf16_nt_fused=gemm
        )
        out = torch.empty((4, 2), dtype=torch.bfloat16)
        with (
            patch.object(entrypoint, "_is_sail", True),
            patch.object(entrypoint, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(entrypoint, "deep_gemm", vendor, create=True),
            patch.object(
                compile_utils, "deep_gemm_execution_hook", return_value=nullcontext()
            ),
        ):
            result = inspect.unwrap(ops.grouped_gemm_nt_bf16bf16bf16_fused)(
                a, b, out, ids, config
            )
        torch.testing.assert_close(
            out, torch.tensor([[1, 2], [2, 4], [6, 8], [9, 12]], dtype=torch.bfloat16)
        )
        for actual, expected in zip(result, (rows, inverse, experts)):
            torch.testing.assert_close(actual, expected)
        restored = out[result[1].long()].reshape(2, 2, 2).sum(1)
        torch.testing.assert_close(
            restored, torch.tensor([[3, 6], [15, 20]], dtype=torch.bfloat16)
        )

    def test_fp8_k128_uses_explicit_scale_semantics(self):
        """block scales with K=128 also have a trailing dim of 1, but the warmup must
        still use the block GEMM rather than the channel GEMM."""
        types = compile_utils.DeepGemmKernelType
        q = torch.ones((2, 128)).to(torch.float8_e4m3fn)
        w = torch.ones((2, 128, 128)).to(torch.float8_e4m3fn)
        rows = torch.ones(2, dtype=torch.int32)
        for channel in (False, True):
            expected = (
                types.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL
                if channel
                else types.GROUPED_GEMM_NT_F8F8BF16_FUSED
            )
            ascale = torch.ones((2, 1))
            bscale = torch.ones((2, 128 if channel else 1, 1))
            out = torch.empty((2, 128), dtype=torch.bfloat16)

            def hook(m, n, k, groups, kernel_type):
                self.assertEqual(kernel_type, expected)
                return nullcontext()

            def gemm(lhs, rhs, result, *metadata):
                result.fill_(128)

            vendor = SimpleNamespace(
                moe_align_block_size=lambda *args, **kwargs: None,
                m_grouped_gemm_fp8_fp8_bf16_nt_fused=gemm,
            )
            with (
                self.subTest(channel=channel),
                patch.object(entrypoint, "_is_sail", True),
                patch.object(entrypoint, "ENABLE_JIT_DEEPGEMM", True),
                patch.object(entrypoint, "deep_gemm", vendor, create=True),
                patch.object(compile_utils, "deep_gemm_execution_hook", hook),
            ):
                entrypoint.grouped_gemm_nt_f8f8bf16_fused(
                    (q, ascale),
                    (w, bscale),
                    out,
                    rows,
                    rows,
                    rows,
                    rows,
                    None,
                    perchannel_quant=channel,
                )
            torch.testing.assert_close(out, torch.full_like(out, 128))

    def test_fp4_requires_k_major_scales(self):
        """MN-major uint16 scales from a plain GEMM must not be fed into gather-read."""
        a = torch.empty((3, 64), dtype=torch.uint8)
        b = torch.empty((2, 128, 64), dtype=torch.uint8)
        c = torch.empty((3, 128), dtype=torch.bfloat16)
        metadata = torch.zeros(3, dtype=torch.int32)
        vendor = SimpleNamespace(
            moe_align_block_size=lambda *a, **kw: None,
            m_grouped_gemm_fp4_fp4_bf16_nt_fused=lambda *a, **kw: None,
        )
        for scale in (
            torch.empty((2, 3), dtype=torch.uint16).t(),
            torch.empty((3, 2), dtype=torch.uint8),
        ):
            with (
                self.subTest(dtype=scale.dtype, stride=scale.stride()),
                patch.object(entrypoint, "_is_sail", True),
                patch.object(entrypoint, "ENABLE_JIT_DEEPGEMM", True),
                patch.object(entrypoint, "deep_gemm", vendor, create=True),
                self.assertRaisesRegex(ValueError, "K-major"),
            ):
                entrypoint.grouped_gemm_nt_f4f4bf16_fused(
                    (a, scale),
                    (b, scale),
                    c,
                    metadata,
                    metadata,
                    metadata,
                    metadata,
                    None,
                )

    def test_fp4_warmup_preserves_packed_width_and_scale_bytes(self):
        """The warmup must reuse the packed K and the K-major LHS; already packed
        scales must not be preprocessed twice."""
        executor = compile_utils._GroupedFusedWarmupExecutor_fp4.__new__(
            compile_utils._GroupedFusedWarmupExecutor_fp4
        )
        executor.max_m = 3
        executor.lhs_q = torch.full((3, 64), 0x22, dtype=torch.uint8)
        executor.lhs_s = torch.full((3, 2), 0x7F7F, dtype=torch.uint16)
        executor.rhs_q = torch.full((2, 128, 64), 0x22, dtype=torch.uint8)
        executor.rhs_s = torch.full((2, 128, 2), 0x7F7F, dtype=torch.uint16)
        executor.topk_ids = torch.zeros((3, 1), dtype=torch.int32)
        executor.out = torch.zeros((3, 128), dtype=torch.bfloat16)
        routes = torch.zeros(2, dtype=torch.int32)
        configs = [16, 128, 128]
        once = iter([(configs, routes, routes, routes, routes, routes, routes)])

        def align(a, b, ids, **kw):
            self.assertEqual(
                (tuple(a.shape), tuple(b.shape), tuple(ids.shape)),
                ((2, 64), (2, 128, 64), (2, 1)),
            )
            return next(once)

        def gemm(lhs, rhs, out, *metadata):
            self.assertEqual(lhs[1].stride(), (2, 1))
            self.assertTrue(
                torch.equal(
                    lhs[1].view(torch.uint8),
                    torch.full((2, 4), 0x7F, dtype=torch.uint8),
                )
            )
            self.assertIs(rhs[1], executor.rhs_s)
            self.assertIs(metadata[-1], configs)
            out.fill_(128)

        vendor = SimpleNamespace(
            moe_align_block_size=align, m_grouped_gemm_fp4_fp4_bf16_nt_fused=gemm
        )
        with patch.object(compile_utils, "deep_gemm", vendor, create=True):
            executor.execute(2)
        torch.testing.assert_close(
            executor.out[:2], torch.full_like(executor.out[:2], 128)
        )
        self.assertEqual(executor.out[2].count_nonzero().item(), 0)

    def test_fp4_warmup_budget_covers_scale_preprocessing_peak(self):
        """With a packed byte K, the budget must cover the uint16 LHS, the RHS raw
        data and the RHS conversion output."""
        # Both m < experts and unaligned m need routing metadata, so computing
        # the GEMM tensors alone is not enough.
        for m, experts in ((1, 256), (257, 8)):
            n, packed_k = 128, 64
            required = compile_utils._BaseWarmupExecutor.get_memory_requirement(
                compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED,
                m,
                n,
                packed_k,
                experts,
            ) * (1 << 30)
            tensors = (
                torch.empty((m, packed_k), dtype=torch.uint8),
                torch.empty((experts, n, packed_k), dtype=torch.uint8),
                torch.empty((m, packed_k // 32), dtype=torch.uint16),
                torch.empty((experts, n, packed_k // 16), dtype=torch.uint8),
                torch.empty((experts, n, packed_k // 32), dtype=torch.uint16),
                torch.empty((m, n), dtype=torch.bfloat16),
                torch.empty((m, 1), dtype=torch.int32),
            )
            allocated = sum(t.numel() * t.element_size() for t in tensors)
            self.assertGreater(required, allocated)

    def test_registered_fake_outputs_cover_topk_expansion(self):
        """The three outputs of the real torch schema must carry E counts and M*topk
        indices."""
        a = torch.empty((3, 128), device="meta", dtype=torch.bfloat16)
        b = torch.empty((5, 256, 128), device="meta", dtype=torch.bfloat16)
        c = torch.empty((6, 256), device="meta", dtype=torch.bfloat16)
        ids = torch.empty((3, 2), device="meta", dtype=torch.int32)
        for name in (
            "grouped_gemm_nt_bf16bf16bf16_fused",
            "grouped_gemm_nt_bf16i4bf16_fused",
            "grouped_gemm_nt_f4f4bf16_fused",
            "grouped_gemm_nt_f8f8bf16_fused",
            "grouped_gemm_nt_i8i8bf16_fused",
        ):
            with self.subTest(name=name):
                op = getattr(torch.ops.sglang, name)
                if "bf16bf16bf16" in name:
                    outputs = op(a, b, c, ids)
                elif "bf16i4bf16" in name:
                    weight = torch.empty((5, 256, 64), device="meta", dtype=torch.uint8)
                    scale = torch.empty((5, 4, 256), device="meta", dtype=torch.uint8)
                    outputs = op(a, weight, scale, c, ids)
                else:
                    scale = torch.empty((3, 1), device="meta")
                    args = (a, scale, b, scale, c, ids)
                    outputs = op(*args, False) if "f8f8" in name else op(*args)
                self.assertEqual([tuple(x.shape) for x in outputs], [(5,), (6,), (6,)])
                self.assertTrue(
                    all(
                        x.dtype == torch.int32 and x.device.type == "meta"
                        for x in outputs
                    )
                )

    def test_w4a16_fused_routes_use_logical_dimensions(self):
        """VALU/INT4 Marlin dims must not be mixed with the MMA packed K; the routing
        is generated only once."""
        types = compile_utils.DeepGemmKernelType
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16).repeat(1, 32)
        ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        tokens = torch.tensor([1, 2, 0, 3], dtype=torch.int32)
        experts = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        rows = torch.tensor([2, 2], dtype=torch.int32)
        inverse = torch.tensor([2, 0, 1, 3], dtype=torch.int32)
        blocks = torch.tensor([4], dtype=torch.int32)
        config = [16, 128, 64]
        for variant, dtype, scale_dtype, kernel_type in (
            (
                "mma",
                torch.uint8,
                torch.uint8,
                types.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED,
            ),
            (
                "valu",
                torch.int32,
                torch.uint8,
                types.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED,
            ),
            (
                "int4",
                torch.int32,
                torch.bfloat16,
                types.GROUPED_GEMM_NT_BF16I4BF16_FUSED,
            ),
        ):
            weight = torch.empty(
                (2, 128, 32) if dtype == torch.uint8 else (2, 4, 256), dtype=dtype
            )
            scales = torch.empty((2, 2, 128), dtype=scale_dtype)
            out = torch.empty((4, 128), dtype=torch.bfloat16)
            once = iter([(config, rows, experts, tokens, blocks, inverse, experts)])

            def tune(m, n, k, **kw):
                self.assertEqual(
                    (m, n, k, kw),
                    (2, 128, 64, {"num_groups": 2, "nopad": True, "dtype": "int4"}),
                )
                return config

            def align(lhs, rhs, topk, **kw):
                self.assertIs(kw["config"], config)
                return next(once)

            def hook(m, n, k, groups, kind):
                self.assertEqual((m, n, k, groups, kind), (2, 128, 64, 2, kernel_type))
                return nullcontext()

            def gemm(
                lhs, rhs, result, counts, expert_data, token_data, block_data, configs
            ):
                self.assertIs(rhs[0], weight)
                self.assertIs(rhs[1], scales)
                self.assertIs(configs, config)
                result.copy_(
                    torch.stack(
                        [
                            lhs[int(t) // 2].sum().expand(128) * (int(e) + 1)
                            for t, e in zip(token_data, expert_data)
                        ]
                    )
                )

            vendor = SimpleNamespace(
                moe_align_block_size=align, m_grouped_gemm_w4a16_fused=gemm
            )
            with (
                self.subTest(variant=variant),
                patch.object(entrypoint, "_is_sail", True),
                patch.object(entrypoint, "ENABLE_JIT_DEEPGEMM", True),
                patch.object(entrypoint, "deep_gemm", vendor, create=True),
                patch.object(entrypoint.tuner, "get_deep_gemm_config", tune),
                patch.object(compile_utils, "deep_gemm_execution_hook", hook),
            ):
                result = inspect.unwrap(ops.grouped_gemm_nt_bf16i4bf16_fused)(
                    a, weight, scales, out, ids
                )
            for actual, expected in zip(result, (rows, inverse, experts)):
                torch.testing.assert_close(actual, expected)
            restored = out[inverse.long()].view(2, 2, 128).sum(1)
            torch.testing.assert_close(
                restored,
                torch.tensor([288, 672], dtype=torch.bfloat16)
                .unsqueeze(1)
                .expand(2, 128),
            )

    def test_w4a16_factory_allocations_fit_warmup_budget(self):
        """The tensor byte count produced by the real factory must be covered by the
        budget; the MMA K is a logical element width."""
        types = compile_utils.DeepGemmKernelType
        names = (
            "GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED",
            "GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD",
            "GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED",
            "GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED",
            "GROUPED_GEMM_NT_BF16I4BF16_FUSED",
        )
        # Only the device allocator boundary is replaced; shape, factory and
        # budget all run the production logic.
        with ExitStack() as stack:
            for fn in ("empty", "zeros", "randint"):
                allocator = getattr(torch, fn)

                def cpu_alloc(*args, _allocator=allocator, **kwargs):
                    kwargs["device"] = "cpu"
                    return _allocator(*args, **kwargs)

                stack.enter_context(patch.object(torch, fn, cpu_alloc))
            for name in names:
                kind = getattr(types, name)
                executor = compile_utils._BaseWarmupExecutor.create(
                    kind, max_m=3, n=128, k=64, num_groups=2
                )
                allocated = sum(
                    t.numel() * t.element_size()
                    for t in vars(executor).values()
                    if isinstance(t, torch.Tensor)
                )
                required = compile_utils._BaseWarmupExecutor.get_memory_requirement(
                    kind, 3, 128, 64, 2
                ) * (1 << 30)
                with self.subTest(name=name):
                    self.assertGreaterEqual(required, allocated)
                    self.assertEqual(
                        entrypoint._get_w4a16_weight_shape(executor.rhs_q), (2, 128, 64)
                    )
                    self.assertEqual(
                        executor.rhs_s.dtype,
                        torch.bfloat16
                        if name.endswith("_BF16I4BF16_FUSED")
                        else torch.uint8,
                    )
                    if "MMA" in name:
                        self.assertEqual(tuple(executor.rhs_s.shape), (2, 2, 128))

    def test_w4a16_runner_outputs_preserve_mask_and_mma_width(self):
        """Run the real contiguous/masked flow, replacing the GPU GEMM/activation with
        CPU numerics only at the boundary."""
        from sglang.kernels.ops import activation
        from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        for variant in ("mma", "valu", "int4"):
            mma = variant == "mma"
            scale_dtype = torch.bfloat16 if variant == "int4" else torch.uint8
            quant = runner_module.DeepGemmMoeQuantInfo(
                w13_weight=torch.empty(
                    (2, 512, 32) if mma else (2, 4, 1024),
                    dtype=torch.uint8 if mma else torch.int32,
                ),
                w2_weight=torch.empty(
                    (2, 64, 128) if mma else (2, 16, 128),
                    dtype=torch.uint8 if mma else torch.int32,
                ),
                w13_scale=torch.empty(
                    (2, 8, 128) if mma else (2, 2, 512), dtype=scale_dtype
                ),
                w2_scale=torch.empty(
                    (2, 1, 512) if mma else (2, 8, 64), dtype=scale_dtype
                ),
                use_mxfp4_w4a16=variant != "int4",
                use_int4_w4a16=variant == "int4",
            )
            runner = object.__new__(runner_module.DeepGemmRunnerCore)
            runner.config = MoeRunnerConfig(
                activation="situ", top_k=2, gemm1_alpha=4.0, gemm1_clamp_limit=25.0
            )
            runner.swiglu_limit = None
            counts = torch.tensor([1, 2], dtype=torch.int32)

            def gemm(lhs, rhs, out, *metadata, **kw):
                if rhs[0] is quant.w13_weight:
                    self.assertEqual(out.shape[-1], 512)
                    out.fill_(1)
                else:
                    self.assertEqual(lhs.shape[-1], 256)
                    out.copy_(lhs[..., :64])

            def situ(x, out, beta, linear_beta):
                gate, up = x.float().chunk(2, -1)
                out.copy_(
                    beta
                    * torch.tanh(gate / beta)
                    * torch.sigmoid(gate)
                    * linear_beta
                    * torch.tanh(up / linear_beta)
                )
                return out

            def masked_situ(x, out, mask, beta, linear_beta, topk, expected_m):
                self.assertIs(mask, counts)
                self.assertEqual(expected_m, 1)
                out.fill_(42)
                for expert, count in enumerate(mask.tolist()):
                    situ(x[expert, :count], out[expert, :count], beta, linear_beta)
                return out

            with (
                self.subTest(variant=variant),
                patch.object(runner_module, "_is_sail", True),
                patch.object(runner_module, "dispose_tensor"),
                patch.object(
                    runner_module.deep_gemm_wrapper,
                    "grouped_gemm_nt_bf16i4bf16_nopad",
                    gemm,
                ),
                patch.object(
                    runner_module.deep_gemm_wrapper,
                    "grouped_gemm_nt_bf16i4bf16_masked",
                    gemm,
                ),
                patch.object(activation, "situ_and_mul", situ),
                patch.object(activation, "situ_and_mul_masked", masked_situ),
            ):
                contig = runner_module.DeepGemmRunnerInput(
                    torch.ones((3, 64), dtype=torch.bfloat16),
                    None,
                    False,
                    m_indices=torch.zeros(3, dtype=torch.int32),
                )
                out = runner.run(
                    contig,
                    quant,
                    {
                        "all_tokens": 3,
                        "hidden_states_device": torch.device("cpu"),
                        "hidden_states_shape": (3, 64),
                    },
                ).hidden_states
                expected = torch.empty((3, 256), dtype=torch.bfloat16)
                situ(torch.ones((3, 512)), expected, 4.0, 25.0)
                torch.testing.assert_close(out, expected[:, :64])
                masked = runner_module.DeepGemmRunnerInput(
                    torch.ones((2, 3, 64), dtype=torch.bfloat16),
                    None,
                    True,
                    masked_m=counts,
                    expected_m=1,
                )
                masked_out = runner.run(
                    masked, quant, {"hidden_states_device": torch.device("cpu")}
                ).hidden_states
                torch.testing.assert_close(masked_out[0, :1], expected[:1, :64])
                torch.testing.assert_close(masked_out[1, :2], expected[:2, :64])
                for expert, count in enumerate(counts.tolist()):
                    torch.testing.assert_close(
                        masked_out[expert, count:],
                        torch.full_like(masked_out[expert, count:], 42),
                    )

    def test_tp_oai_layout_survives_fused_handoff_and_nopad(self):
        """Both weight layouts of the same gate/up must give identical output after
        the upper-level forwarding, inverse routing and weighting."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner import sail_deepgemm_moe as tp
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        source = torch.ones((2, 4), dtype=torch.bfloat16)
        ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        counts = torch.tensor([2, 2], dtype=torch.int32)
        inverse = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
        experts = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        gate = torch.tensor([[10, -10, 0.5, -1]], dtype=torch.bfloat16).expand(4, -1)
        up = torch.tensor([[-10, 10, -0.5, 2]], dtype=torch.bfloat16).expand(4, -1)
        g = gate.float().clamp(max=7)
        activated = (
            g * torch.sigmoid(g * 1.702) * (up.float().clamp(-7, 7) + 1)
        ).bfloat16()
        down = (activated.float() * (experts + 1).unsqueeze(-1)).bfloat16()
        expected = (down[inverse.long()] * weights.unsqueeze(-1)).sum(1) * 0.5
        quant = tp.DeepGemmMoeQuantInfo(
            w13_weight=torch.empty((2, 8, 4), dtype=torch.bfloat16),
            w2_weight=torch.empty((2, 4, 4), dtype=torch.bfloat16),
        )

        def count(*args):
            return counts, counts

        def scatter(**kw):
            kw["output_tensor"].fill_(1)
            kw["m_indices"].copy_(experts)
            kw["output_index"].copy_(inverse)
            kw["expert_start_loc"].copy_(torch.tensor([0, 2], dtype=torch.int32))

        def gather(**kw):
            restored = kw["input_tensor"][kw["input_index"].long()]
            kw["output_tensor"].copy_(
                (restored * kw["recv_topk_weight"].unsqueeze(-1)).sum(1)
                * kw["routed_scaling_factor"]
            )

        for fused in (False, True):
            for interleaved in (False, True):
                packed = (
                    torch.stack((gate, up), -1).flatten(-2)
                    if interleaved
                    else torch.cat((gate, up), -1)
                )
                stages = []

                def fused_gemm(lhs, rhs, out, topk):
                    self.assertIs(rhs, quant.w13_weight)
                    out.copy_(packed)
                    stages.append("fused1")
                    return counts, inverse.flatten(), experts

                def nopad(lhs, rhs, out, expert_ids, rows):
                    self.assertIs(rows, counts)
                    torch.testing.assert_close(expert_ids, experts)
                    if rhs is quant.w13_weight:
                        out.copy_(packed)
                        stages.append("nopad1")
                    else:
                        out.copy_(lhs.float() * (experts + 1).unsqueeze(-1))
                        stages.append("nopad2")

                with ExitStack() as stack:
                    stack.enter_context(
                        self.subTest(fused=fused, interleaved=interleaved)
                    )
                    stack.enter_context(
                        envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.override(fused)
                    )
                    for name, fn in (
                        ("count_expert_num_tokens", count),
                        ("ep_scatter_sail", scatter),
                        ("ep_gather", gather),
                        ("grouped_gemm_nt_bf16bf16bf16_fused", fused_gemm),
                        ("grouped_gemm_nt_bf16bf16bf16_nopad", nopad),
                    ):
                        stack.enter_context(patch.object(tp, name, fn))
                    # Only the compile boundary is cancelled; the helper's
                    # real computation and layout choice still run.
                    for name in (
                        "swiglu_with_alpha_and_limit",
                        "swiglu_no_interleaved_with_alpha_and_limit",
                    ):
                        stack.enter_context(
                            patch.object(tp, name, inspect.unwrap(getattr(tp, name)))
                        )
                    result = tp.fused_experts_none_to_deep_gemm(
                        SimpleNamespace(
                            hidden_states=source, topk_output=(weights, ids, None)
                        ),
                        quant,
                        MoeRunnerConfig(
                            gemm1_alpha=1.702,
                            gemm1_clamp_limit=7.0,
                            gate_up_interleaved=interleaved,
                            routed_scaling_factor=0.5,
                            inplace=False,
                        ),
                    ).hidden_states
                torch.testing.assert_close(
                    result, expected.bfloat16(), rtol=0.02, atol=0.05
                )
                torch.testing.assert_close(source, torch.ones_like(source))
                self.assertEqual(stages, ["fused1" if fused else "nopad1", "nopad2"])

    def test_w4a16_tp_fused_and_nopad_preserve_weighted_output(self):
        """GEMM2 of the real TP runner must stay nopad and reuse the inverse routing
        of GEMM1."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner import sail_deepgemm_moe as tp

        source = torch.full((2, 64), 0.01, dtype=torch.bfloat16)
        source[1].fill_(0.02)
        ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        tokens = torch.tensor([1, 2, 0, 3], dtype=torch.int32)
        experts = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        inverse = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
        counts = torch.tensor([2, 2], dtype=torch.int32)

        def count(topk, local_experts, alignment):
            self.assertEqual((local_experts, alignment), (2, 1))
            return counts, counts

        def scatter(**kw):
            kw["output_tensor"].copy_(kw["recv_x"][tokens.long() // 2])
            kw["m_indices"].copy_(experts)
            kw["output_index"].copy_(inverse)
            kw["expert_start_loc"].copy_(torch.tensor([0, 2], dtype=torch.int32))

        def situ(x, out, beta, linear_beta):
            if out is None:
                out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype)
            gate, up = x.float().chunk(2, -1)
            out.copy_(
                beta
                * torch.tanh(gate / beta)
                * torch.sigmoid(gate)
                * linear_beta
                * torch.tanh(up / linear_beta)
            )
            return out

        def gather(**kw):
            restored = kw["input_tensor"][kw["input_index"].long()]
            kw["output_tensor"].copy_(
                (restored * kw["recv_topk_weight"].unsqueeze(-1)).sum(1)
                * kw["routed_scaling_factor"]
            )

        for mma in (False, True):
            w1 = torch.empty(
                (2, 512, 32) if mma else (2, 4, 1024),
                dtype=torch.uint8 if mma else torch.int32,
            )
            w2 = torch.empty((2, 64, 128) if mma else (2, 16, 128), dtype=w1.dtype)
            s1 = torch.empty((2, 8, 128) if mma else (2, 2, 512), dtype=torch.uint8)
            s2 = torch.empty((2, 1, 512) if mma else (2, 8, 64), dtype=torch.uint8)
            for fused in (False, True):
                stages = []

                def gemm1(lhs, result):
                    self.assertEqual(tuple(result.shape), (4, 512))
                    result.copy_(
                        (lhs.float().sum(-1) * (experts + 1))
                        .unsqueeze(-1)
                        .expand(4, 512)
                    )

                def fused_gemm(lhs, rhs, scale, result, topk):
                    self.assertIs(rhs, w1)
                    stages.append("fused1")
                    gemm1(lhs[tokens.long() // 2], result)
                    return counts, inverse.flatten(), experts

                def nopad(lhs, rhs, scale, result, expert_ids, rows):
                    self.assertIs(rows, counts)
                    torch.testing.assert_close(expert_ids, experts)
                    if rhs is w1:
                        stages.append("nopad1")
                        gemm1(lhs, result)
                    else:
                        stages.append("nopad2")
                        self.assertIs(rhs, w2)
                        self.assertEqual(tuple(lhs.shape), (4, 256))
                        result.copy_(lhs[:, :64].float() * (experts + 1).unsqueeze(-1))

                actual = torch.empty_like(source)
                with (
                    self.subTest(mma=mma, fused=fused),
                    envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.override(fused),
                    patch.object(tp, "count_expert_num_tokens", count),
                    patch.object(tp, "ep_scatter_sail", scatter),
                    patch.object(tp, "grouped_gemm_nt_bf16i4bf16_fused", fused_gemm),
                    patch.object(tp, "grouped_gemm_nt_bf16i4bf16_nopad", nopad),
                    patch.object(tp, "situ_and_mul", situ),
                    patch.object(tp, "ep_gather", gather),
                ):
                    tp.deep_moe_impl_fused(
                        source.clone(),
                        w1,
                        w2,
                        s1,
                        s2,
                        weights,
                        ids,
                        use_mxfp4_w4a16=True,
                        gemm1_alpha=4.0,
                        gemm1_limit=25.0,
                        activation="situ",
                        routed_scaling_factor=0.5,
                        out_hidden_states=actual,
                    )
                # Compute independently over unpermuted token/expert pairs;
                # both GEMMs round at the BF16 output.
                projected = (
                    (source.float().sum(-1, keepdim=True) * (ids + 1))
                    .to(torch.bfloat16)
                    .float()
                )
                activated = (
                    4.0
                    * torch.tanh(projected / 4.0)
                    * torch.sigmoid(projected)
                    * 25.0
                    * torch.tanh(projected / 25.0)
                ).to(torch.bfloat16)
                down = (activated.float() * (ids + 1)).to(torch.bfloat16)
                expected = (down * weights).sum(1, keepdim=True) * 0.5
                torch.testing.assert_close(
                    actual, expected.to(torch.bfloat16).expand(2, 64)
                )
                self.assertEqual(stages, ["fused1" if fused else "nopad1", "nopad2"])

    def test_deepep_ll_handoff_preserves_mask_and_state(self):
        """LL is already a masked layout, so BF16 W4A16 must not be requantized and
        the routing and capacity must not be lost."""
        from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        ids = torch.tensor([[1, 0]], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75]])
        counts = torch.tensor([1, 1], dtype=torch.int32)
        for dtype, scale_width, group in (
            (torch.bfloat16, None, 128),
            (torch.uint8, 2, 32),
            (torch.float8_e4m3fn, 2, 32),
        ):
            x = torch.empty((2, 3, 64), dtype=dtype)
            scale = None if scale_width is None else torch.empty((2, 3, scale_width))
            state = {}
            with patch.object(runner_module, "_is_sail", True):
                payload = runner_module.pre_permute_deepep_ll_to_deep_gemm(
                    (x, scale, ids, weights, counts, 1),
                    None,
                    MoeRunnerConfig(top_k=2),
                    state,
                )
            self.assertIs(payload.hidden_states, x)
            self.assertIs(payload.hidden_states_scale, scale)
            self.assertIs(payload.masked_m, counts)
            self.assertTrue(payload.use_masked_gemm)
            self.assertEqual(
                (payload.expected_m, payload.activation_scale_block_size), (1, group)
            )
            self.assertIs(state["topk_ids"], ids)
            self.assertIs(state["topk_weights"], weights)
            self.assertEqual(
                (
                    state["hidden_states_shape"],
                    state["hidden_states_dtype"],
                    state["hidden_states_device"],
                ),
                (x.shape, x.dtype, x.device),
            )

    def test_w4a4_situ_runner_preserves_mn_major_scale_views(self):
        """The physical scale of the SiTU producer must still be an MN-major view for
        GEMM2 after t/permute."""
        from sglang.kernels.ops import activation
        from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        quant = runner_module.DeepGemmMoeQuantInfo(
            w13_weight=torch.empty((2, 512, 32), dtype=torch.uint8),
            w2_weight=torch.empty((2, 64, 128), dtype=torch.uint8),
            w13_scale=torch.empty((2, 512, 1), dtype=torch.uint16),
            w2_scale=torch.empty((2, 64, 4), dtype=torch.uint16),
            use_mxfp4=True,
        )
        runner = object.__new__(runner_module.DeepGemmRunnerCore)
        runner.config = MoeRunnerConfig(
            activation="situ", top_k=2, gemm1_clamp_limit=None
        )
        runner.swiglu_limit = None
        counts = torch.tensor([1, 2], dtype=torch.int32)
        produced_scales = []

        def produce(x, output, scale, beta, linear_beta):
            self.assertEqual((beta, linear_beta), (4.0, None))
            self.assertEqual(output.shape[-1], 128)
            self.assertEqual(scale.dtype, torch.uint16)
            output.fill_(0x22)
            scale.copy_(
                torch.arange(scale.numel()).reshape(scale.shape).to(torch.uint16)
            )
            produced_scales.append(scale)

        def produce_masked(x, output, scale, mask, beta, linear_beta, topk, expected_m):
            self.assertIs(mask, counts)
            self.assertEqual((topk, expected_m), (2, 1))
            produce(x, output, scale, beta, linear_beta)

        def gemm(lhs, rhs, bias, out, *metadata, **kw):
            if rhs[0] is quant.w13_weight:
                out.fill_(1)
            else:
                q, scale = lhs
                physical = produced_scales[-1]
                expected = physical.permute(0, 2, 1) if out.ndim == 3 else physical.t()
                self.assertEqual(scale.stride(), expected.stride())
                self.assertEqual(scale.data_ptr(), physical.data_ptr())
                torch.testing.assert_close(scale, expected)
                torch.testing.assert_close(q, torch.full_like(q, 0x22))
                out.fill_(2)
            return (16, 1)

        with (
            patch.object(runner_module, "_is_sail", True),
            patch.object(runner_module, "dispose_tensor"),
            patch.dict(
                sys.modules,
                {"deep_gemm": SimpleNamespace(preprocess_mxfp4_scales=lambda x: x)},
            ),
            patch.object(
                runner_module.deep_gemm_wrapper, "grouped_gemm_nt_f4f4bf16_nopad", gemm
            ),
            patch.object(
                runner_module.deep_gemm_wrapper, "grouped_gemm_nt_f4f4bf16_masked", gemm
            ),
            patch.object(activation, "situ_and_mul_post_quant_mxfp4", produce),
            patch.object(
                activation, "situ_and_mul_masked_post_quant_mxfp4", produce_masked
            ),
        ):
            for masked in (False, True):
                shape = (2, 3, 32) if masked else (3, 32)
                payload = runner_module.DeepGemmRunnerInput(
                    torch.empty(shape, dtype=torch.uint8),
                    torch.empty((*shape[:-1], 1), dtype=torch.uint16),
                    masked,
                    masked_m=counts if masked else None,
                    expected_m=1,
                    m_indices=torch.zeros(3, dtype=torch.int32),
                )
                state = {
                    "all_tokens": 3,
                    "hidden_states_shape": (3, 32),
                    "hidden_states_dtype": torch.uint8,
                    "hidden_states_device": torch.device("cpu"),
                    "meta_overlap_args": {},
                }
                with self.subTest(masked=masked):
                    result = runner.run(payload, quant, state).hidden_states
                    self.assertEqual(tuple(result.shape), (*shape[:-1], 64))
                    torch.testing.assert_close(result, torch.full_like(result, 2))
                    if masked:
                        self.assertEqual(
                            state["meta_overlap_args"], {"block_m": 16, "threshold": 1}
                        )


if __name__ == "__main__":
    unittest.main()
