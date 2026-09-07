import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.runtime_context import get_parallel, override_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Collectives are mocked and world_size is a plain int, so the world_size=4
# cases need one real CUDA device.
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _FakeWorkspace:
    def __init__(self, backend, world_size, dtype=torch.bfloat16):
        self.backend = backend
        self.world_size = world_size
        self.metadata = {"use_fp32_lamport": dtype == torch.float32}

    def is_buffer_size_sufficient(self, **_kwargs):
        return True


class _FakeFlashInferComm:
    class AllReduceFusionPattern:
        kAllReduce = object()
        kARResidualRMSNorm = object()

    def __init__(self):
        self.calls = []
        self.fusion_kwargs = None

    def create_allreduce_fusion_workspace(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeWorkspace(
            kwargs["backend"], kwargs["world_size"], dtype=kwargs["dtype"]
        )

    def allreduce_fusion(
        self,
        *,
        input,
        workspace,
        pattern,
        output=None,
        residual_out=None,
        norm_out=None,
        residual_in=None,
        rms_gamma=None,
        rms_eps=None,
        **_kwargs,
    ):
        self.fusion_kwargs = _kwargs
        if pattern is self.AllReduceFusionPattern.kAllReduce:
            allreduced = input * workspace.world_size
            if output is None:
                return allreduced
            output.copy_(allreduced)
            return output

        if pattern is not self.AllReduceFusionPattern.kARResidualRMSNorm:
            raise ValueError(f"Unexpected pattern: {pattern}")

        allreduced = input * workspace.world_size
        expected_residual = allreduced + residual_in
        variance = expected_residual.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        expected_norm = (
            expected_residual.to(torch.float32)
            * torch.rsqrt(variance + rms_eps)
            * rms_gamma.to(torch.float32)
        ).to(input.dtype)
        residual_out.copy_(expected_residual)
        norm_out.copy_(expected_norm)


class _FakePureMoeAllReduceAPI:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        world_size,
        world_rank,
        token_num,
        hidden_dim,
        workspace_ptrs,
        launch_with_pdl,
        residual_in,
        rms_gamma,
        rms_eps,
        scale_factor,
        moe_reduction_device_num_experts,
        moe_reduction_scale_input,
        moe_reduction_active_experts_token_input,
        moe_reduction_token_input,
        layout_code,
        moe_allreduce_out,
        residual_out,
        norm_out,
        quant_out,
        scale_out,
        weight_bias=None,
        *,
        backend="trtllm",
    ):
        call = locals().copy()
        call.pop("self")
        self.calls.append(call)
        residual_out.copy_(residual_in + 1)
        norm_out.copy_(residual_in + 2)


def _torch_allreduce_residual_rmsnorm_baseline(
    input_tensor, residual, weight, world_size, eps
):
    allreduced = input_tensor * world_size
    residual_out = allreduced + residual
    variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    norm_out = (
        residual_out.to(torch.float32)
        * torch.rsqrt(variance + eps)
        * weight.to(torch.float32)
    ).to(input_tensor.dtype)
    return norm_out, residual_out


class TestFlashInferTrtllmMoeAllReduce(CustomTestCase):
    def test_resolver_requires_the_exact_22_parameter_api(self):
        api = _FakePureMoeAllReduceAPI()

        class PureOnlyComm:
            trtllm_moe_allreduce_fusion = api

            def __getattr__(self, name):
                raise AssertionError(f"the pure resolver inspected unexpected {name}")

        self.assertIs(
            fusion._get_flashinfer_trtllm_moe_allreduce_api(PureOnlyComm()), api
        )

        def missing_backend(*args, **kwargs):
            return None

        self.assertIsNone(
            fusion._get_flashinfer_trtllm_moe_allreduce_api(
                SimpleNamespace(trtllm_moe_allreduce_fusion=missing_backend)
            )
        )

    def test_payload_admission_uses_lamport_workspace_byte_limit(self):
        for world_size, max_tokens in ((2, 74825), (4, 37412), (8, 18706)):
            with self.subTest(world_size=world_size):
                self.assertTrue(
                    fusion._cake_moe_allreduce_lamport_workspace_supported(
                        2049, 7168, world_size
                    )
                )
                self.assertTrue(
                    fusion._cake_moe_allreduce_lamport_workspace_supported(
                        max_tokens, 7168, world_size
                    )
                )
                self.assertFalse(
                    fusion._cake_moe_allreduce_lamport_workspace_supported(
                        max_tokens + 1, 7168, world_size
                    )
                )

        self.assertFalse(
            fusion._cake_moe_allreduce_lamport_workspace_supported(0, 7168, 8)
        )

    def test_layout_adapter_produces_expert_major_inputs(self):
        gemm2_out = torch.arange(15, dtype=torch.bfloat16).reshape(5, 3)
        expert_weights = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16
        )
        expanded_idx = torch.tensor([[2, 0], [4, 1]], dtype=torch.int32)

        active, scales = fusion.materialize_flashinfer_trtllm_moe_allreduce_layout(
            gemm2_out, expert_weights, expanded_idx
        )

        expected_active = torch.stack(
            (
                torch.stack((gemm2_out[2], gemm2_out[4])),
                torch.stack((gemm2_out[0], gemm2_out[1])),
            )
        )
        torch.testing.assert_close(active, expected_active)
        torch.testing.assert_close(
            scales, expert_weights.transpose(0, 1).to(torch.float32)
        )
        self.assertEqual(active.shape, (2, 2, 3))
        self.assertEqual(scales.dtype, torch.float32)
        self.assertTrue(active.is_contiguous())
        self.assertTrue(scales.is_contiguous())

    def test_dispatch_passes_exact_api_and_cake_backend(self):
        api = _FakePureMoeAllReduceAPI()
        payload = fusion.FlashInferTrtllmMoeAllReducePayload(
            active_experts_token_input=torch.randn(2, 3, 4),
            scale_input=torch.randn(2, 3, dtype=torch.float32),
            token_input=torch.randn(3, 4),
            workspace_ptrs=torch.zeros(13, dtype=torch.int64),
            world_rank=1,
            world_size=4,
            launch_with_pdl=True,
        )
        residual = torch.randn(3, 4)
        norm_weight = torch.randn(4)

        with patch.object(fusion, "_flashinfer_trtllm_moe_allreduce", api):
            norm_out, residual_out = fusion.run_flashinfer_trtllm_moe_allreduce(
                payload, residual, norm_weight, 1e-6
            )

        self.assertEqual(len(api.calls), 1)
        call = api.calls[0]
        self.assertEqual(tuple(call), fusion._TRTLLM_MOE_ALLREDUCE_REQUIRED_PARAMS)
        self.assertEqual(call["backend"], "cake")
        self.assertIs(call["moe_reduction_token_input"], payload.token_input)
        self.assertIsNone(call["moe_allreduce_out"])
        self.assertIsNone(call["quant_out"])
        self.assertIsNone(call["scale_out"])
        torch.testing.assert_close(residual_out, residual + 1)
        torch.testing.assert_close(norm_out, residual + 2)

    def test_missing_pure_api_falls_back_before_layout_materialization(self):
        materialize = MagicMock()
        with (
            patch.object(fusion, "_flashinfer_trtllm_moe_allreduce", None),
            patch.object(
                fusion,
                "materialize_flashinfer_trtllm_moe_allreduce_layout",
                materialize,
            ),
        ):
            result = fusion.prepare_flashinfer_trtllm_moe_allreduce_payload(
                gemm2_out=torch.empty(16, 8),
                expert_weights=torch.empty(2, 8),
                expanded_idx_to_permuted_idx=torch.empty(2, 8, dtype=torch.int32),
                shared_expert_output=torch.empty(2, 8),
                top_k=8,
                launch_with_pdl=True,
            )

        self.assertIsNone(result)
        materialize.assert_not_called()

    def test_decoder_consumes_payload_only_once(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        layer = object.__new__(DeepseekV2DecoderLayer)
        object.__setattr__(
            layer,
            "input_layernorm",
            SimpleNamespace(weight=torch.randn(4), variance_epsilon=1e-6),
        )
        payload = fusion.FlashInferTrtllmMoeAllReducePayload(
            active_experts_token_input=torch.randn(2, 3, 4),
            scale_input=torch.randn(2, 3, dtype=torch.float32),
            token_input=torch.randn(3, 4),
            workspace_ptrs=torch.zeros(13, dtype=torch.int64),
            world_rank=0,
            world_size=4,
            launch_with_pdl=True,
        )
        residual = torch.randn(3, 4)
        norm_out = torch.randn(3, 4)
        residual_out = torch.randn(3, 4)

        with patch.object(
            fusion,
            "run_flashinfer_trtllm_moe_allreduce",
            return_value=(norm_out, residual_out),
        ) as run:
            first_hidden, first_residual, consumed = (
                layer._consume_flashinfer_trtllm_moe_allreduce(payload, residual)
            )
            second_hidden, second_residual, consumed_again = (
                layer._consume_flashinfer_trtllm_moe_allreduce(
                    first_hidden, first_residual
                )
            )

        self.assertTrue(consumed)
        self.assertFalse(consumed_again)
        self.assertIs(first_hidden, norm_out)
        self.assertIs(first_residual, residual_out)
        self.assertIs(second_hidden, norm_out)
        self.assertIs(second_residual, residual_out)
        run.assert_called_once()


class TestFlashInferCommFusion(CustomTestCase):
    """The arch dispatch is `_resolve_backend(backend, is_multi_node)`.

    The public entry above it takes no arguments -- it reads
    `exec.comm.flashinfer_allreduce_fusion_backend` and `parallel.nnodes` off the
    published bags -- so the cases here drive the dispatch directly.
    """

    def test_auto_backend_resolves_by_arch(self):
        single_node = ("auto", False)
        multi_node = ("auto", True)

        # Blackwell: mnnvl on both single-node and multi-node.
        with override_platform(is_sm100=True):
            self.assertEqual(
                fusion._resolve_backend(*single_node),
                "mnnvl",
            )
            self.assertEqual(fusion._resolve_backend(*multi_node), "mnnvl")

        # SM90: auto uses trtllm on single-node, multi-node is unsupported.
        with (
            override_platform(is_sm100=False),
            override_platform(is_sm90=True),
        ):
            self.assertEqual(
                fusion._resolve_backend(*single_node),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion._resolve_backend(*multi_node)

        # Architectures outside SM90/SM10X are unsupported. Both pre-SM90
        # and post-SM10X devices (e.g. SM120) must fail closed.
        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                override_platform(is_sm100=False),
                override_platform(is_sm90=False),
            ):
                with self.assertRaises(ValueError):
                    fusion._resolve_backend(*single_node)
                with self.assertRaises(ValueError):
                    fusion._resolve_backend(*multi_node)

    def test_explicit_backend_validation(self):
        single_node_mnnvl = ("mnnvl", False)
        multi_node_mnnvl = ("mnnvl", True)
        single_node_trtllm = ("trtllm", False)
        multi_node_trtllm = ("trtllm", True)

        with (
            override_platform(is_sm100=False),
            override_platform(is_sm90=True),
        ):
            self.assertEqual(
                fusion._resolve_backend(*single_node_mnnvl),
                "mnnvl",
            )
            self.assertEqual(
                fusion._resolve_backend(*single_node_trtllm),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion._resolve_backend(*multi_node_mnnvl)
            with self.assertRaises(ValueError):
                fusion._resolve_backend(*multi_node_trtllm)

        with override_platform(is_sm100=True):
            self.assertEqual(
                fusion._resolve_backend(*multi_node_mnnvl),
                "mnnvl",
            )
            with self.assertRaises(ValueError):
                fusion._resolve_backend(*multi_node_trtllm)

        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                override_platform(is_sm100=False),
                override_platform(is_sm90=False),
            ):
                for args in (
                    single_node_mnnvl,
                    multi_node_mnnvl,
                    single_node_trtllm,
                    multi_node_trtllm,
                ):
                    with self.subTest(backend=args[0], multi_node=args[1]):
                        with self.assertRaises(ValueError):
                            fusion._resolve_backend(*args)

    def test_allreduce_fusion_backends_match_torch_baseline(self):
        fake_comm = _FakeFlashInferComm()
        original_comm = fusion._flashinfer_comm
        original_create = fusion._create_allreduce_fusion_workspace
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        manager_key = "flashinfer_fusion_attn_tp_workspace"
        original_manager = buffers.get(manager_key)
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._create_allreduce_fusion_workspace = (
                fake_comm.create_allreduce_fusion_workspace
            )
            fusion._flashinfer_allreduce_unavailable = False

            for backend in ("trtllm", "mnnvl"):
                with self.subTest(backend=backend):
                    world_size = 4
                    manager = fusion.FlashInferWorkspaceManager()
                    manager.workspace = _FakeWorkspace(backend, world_size)
                    manager.backend = backend
                    manager.initialized = True
                    buffers[manager_key] = manager
                    if not torch.cuda.is_available():
                        self.skipTest("FlashInfer allreduce custom op is CUDA-only")
                    device = torch.device("cuda")
                    torch.manual_seed(0)
                    input_tensor = torch.randn(4, 8, dtype=torch.float32, device=device)
                    residual = torch.randn(4, 8, dtype=torch.float32, device=device)
                    weight = torch.randn(8, dtype=torch.float32, device=device)
                    eps = 1e-6

                    expected_norm, expected_residual = (
                        _torch_allreduce_residual_rmsnorm_baseline(
                            input_tensor, residual, weight, world_size, eps
                        )
                    )

                    with (
                        patch.object(
                            fusion, "is_flashinfer_available", return_value=True
                        ),
                        get_parallel().override(attn_tp_size=world_size),
                        patch.object(
                            fusion, "ensure_workspace_initialized", return_value=True
                        ),
                    ):
                        norm_out, residual_out = (
                            fusion.flashinfer_allreduce_residual_rmsnorm(
                                input_tensor=input_tensor,
                                residual=residual,
                                weight=weight,
                                eps=eps,
                                max_token_num=8,
                            )
                        )

                    torch.testing.assert_close(norm_out, expected_norm)
                    torch.testing.assert_close(residual_out, expected_residual)
                    self.assertEqual(
                        fake_comm.fusion_kwargs.get("fp32_acc", False),
                        backend == "trtllm",
                    )
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._create_allreduce_fusion_workspace = original_create
            if original_manager is None:
                buffers.pop(manager_key, None)
            else:
                buffers[manager_key] = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable


_GROUP_KEY = ("device_group", "cpu_group")
_OTHER_GROUP_KEY = ("other_device_group", "other_cpu_group")


class TestFlashInferAllReduceOnly(CustomTestCase):
    def _make_manager(self, world_size, group_key=_GROUP_KEY, backend="trtllm"):
        manager = fusion.FlashInferWorkspaceManager()
        manager.workspace = _FakeWorkspace(backend, world_size)
        manager.initialized = True
        manager.world_size = world_size
        manager.group = group_key
        manager.max_token_num = 2048
        manager.hidden_dim = 4096
        manager.dtype = torch.float32
        manager.backend = backend
        manager.use_fp32_lamport = True
        return manager

    @contextlib.contextmanager
    def _patched_attn_workspace(self, manager):
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        manager_key = "flashinfer_fusion_attn_tp_workspace"
        original_manager = buffers.get(manager_key)
        original_comm = fusion._flashinfer_comm
        original_unavailable = fusion._flashinfer_allreduce_unavailable

        buffers[manager_key] = manager
        fake_comm = _FakeFlashInferComm()
        fusion._flashinfer_comm = fake_comm
        fusion._flashinfer_allreduce_unavailable = False
        try:
            yield fake_comm
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._flashinfer_allreduce_unavailable = original_unavailable
            if original_manager is None:
                buffers.pop(manager_key, None)
            else:
                buffers[manager_key] = original_manager

    def _can_use(self, input_, world_size=4, group_key=_GROUP_KEY):
        return fusion.can_use_flashinfer_allreduce(
            input_,
            use_attn_tp_group=True,
            expected_world_size=world_size,
            expected_group_key=group_key,
        )

    def test_allreduce_output_equals_input_times_world_size(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for flashinfer custom op")
        world_size = 4
        manager = self._make_manager(world_size)
        manager.dtype = torch.bfloat16
        manager.use_fp32_lamport = False
        with self._patched_attn_workspace(manager) as fake_comm:
            input_ = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
            expected = input_ * world_size

            with get_parallel().override(attn_tp_size=world_size):
                self.assertTrue(self._can_use(input_, world_size=world_size))
                result = fusion.flashinfer_allreduce(input_, use_attn_tp_group=True)

            torch.testing.assert_close(result, expected)
            # trtllm rounds to the input dtype on every rank without this
            self.assertTrue(fake_comm.fusion_kwargs["fp32_acc"])

    def test_shape_guard_rejects_non_2d(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            self.assertFalse(self._can_use(torch.randn(16)))
            self.assertFalse(self._can_use(torch.randn(2, 8, 16)))

    def test_shape_guard_rejects_non_contiguous(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            non_contiguous = torch.randn(16, 8).t()
            self.assertFalse(non_contiguous.is_contiguous())
            self.assertFalse(self._can_use(non_contiguous))

    def test_rejects_when_unavailable(self):
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_allreduce_unavailable = True
            self.assertFalse(self._can_use(torch.randn(8, 16)))
        finally:
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_rejects_when_workspace_uninitialized(self):
        with self._patched_attn_workspace(fusion.FlashInferWorkspaceManager()):
            with get_parallel().override(attn_tp_size=4):
                self.assertFalse(self._can_use(torch.randn(8, 16)))

    def test_rejects_when_workspace_group_differs(self):
        """A workspace rendezvoused on other peers must not be reused.

        Under hybrid EP+TP (e.g. tp=4, ep=2) the MoE-TP and MoE-EP groups have
        the same world size but pair different ranks, so a workspace built for
        one silently reduces across the wrong peers when used by the other --
        wrong output rather than a crash.
        """
        with self._patched_attn_workspace(self._make_manager(2)):
            self.assertFalse(
                self._can_use(
                    torch.randn(8, 16), world_size=2, group_key=_OTHER_GROUP_KEY
                )
            )

    def test_rejects_when_workspace_world_size_differs(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            self.assertFalse(self._can_use(torch.randn(8, 16), world_size=2))

    def test_fp32_initialization_caches_allocated_lamport_mode(self):
        """FP32 startup allocation must remain eligible for FP32 all-reduce.

        The initialization API's legacy use_fp32_lamport argument defaults to
        False, while FlashInfer derives the allocated TRT-LLM mode from dtype.
        Eligibility must follow the workspace metadata rather than that input.
        """
        fake_comm = _FakeFlashInferComm()
        manager = fusion.FlashInferWorkspaceManager()
        with (
            patch.object(fusion, "_flashinfer_comm", fake_comm),
            patch.object(
                fusion,
                "_create_allreduce_fusion_workspace",
                fake_comm.create_allreduce_fusion_workspace,
            ),
            patch.object(
                fusion, "_preflight_check_workspace_memory", return_value=True
            ),
        ):
            manager.initialize(
                world_size=4,
                rank=0,
                max_token_num=8,
                hidden_dim=4096,
                backend="trtllm",
                dtype=torch.float32,
            )

        self.assertTrue(manager.use_fp32_lamport)
        with self._patched_attn_workspace(manager):
            self.assertTrue(
                self._can_use(
                    torch.randn(8, 16, dtype=torch.float32), group_key=(None, None)
                )
            )

    def test_rejects_when_token_num_exceeds_workspace_capacity(self):
        """Oversized all-reduces fall back without triggering a warning.

        TRT-LLM warns whenever its size validator rejects an operation. The
        total element count already proves that this operation cannot use the
        workspace, so the validator must not be invoked.
        """
        manager = self._make_manager(4)
        manager.max_token_num = 8
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=True)
        with self._patched_attn_workspace(manager):
            self.assertFalse(self._can_use(torch.randn(9, 4096)))

        manager.workspace.is_buffer_size_sufficient.assert_not_called()

    def test_reshaped_input_within_total_capacity_reaches_validator(self):
        manager = self._make_manager(4)
        manager.max_token_num = 8
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=True)
        input_ = torch.randn(9, 16)
        with self._patched_attn_workspace(manager):
            self.assertTrue(self._can_use(input_))

        manager.workspace.is_buffer_size_sufficient.assert_called_once_with(
            tp_size=4,
            num_tokens=9,
            hidden_dim=16,
            dtype=input_.dtype,
        )

    def test_non_fp32_dtype_change_reaches_validator(self):
        manager = self._make_manager(4)
        manager.dtype = torch.bfloat16
        manager.use_fp32_lamport = False
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=True)
        input_ = torch.randn(8, 16, dtype=torch.float16)
        with self._patched_attn_workspace(manager):
            self.assertTrue(self._can_use(input_))

        manager.workspace.is_buffer_size_sufficient.assert_called_once()

    def test_mnnvl_capacity_decision_reaches_validator(self):
        manager = self._make_manager(4, backend="mnnvl")
        manager.max_token_num = 8
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=False)
        with self._patched_attn_workspace(manager):
            self.assertFalse(self._can_use(torch.randn(9, 4096)))

        manager.workspace.is_buffer_size_sufficient.assert_called_once()

    def test_compiling_rejects_when_token_num_exceeds_workspace_capacity(self):
        manager = self._make_manager(4)
        manager.max_token_num = 8
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16)))
                self.assertFalse(self._can_use(torch.randn(9, 16)))

    def test_rejects_when_hidden_dim_exceeds_workspace_capacity(self):
        manager = self._make_manager(4)
        manager.hidden_dim = 16
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16)))
                self.assertFalse(self._can_use(torch.randn(8, 17)))

    def test_rejects_when_dtype_mismatches_workspace(self):
        manager = self._make_manager(4)
        manager.dtype = torch.bfloat16
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16, dtype=torch.bfloat16)))
                self.assertFalse(self._can_use(torch.randn(8, 16, dtype=torch.float32)))


class _FakeGroupCoordinator:
    def __init__(self, world_size):
        self.world_size = world_size
        self._fi_workspace_hint = None


class TestTagGroupsForFlashInferAllReduceOnly(CustomTestCase):
    """The MoE workspace rendezvouses on the EP group when moe_ep_size > 1 and
    on the MoE-TP group otherwise, so only that one group may be tagged."""

    def _tag(self, *, attn_tp, moe_ep, moe_tp):
        from sglang.srt.distributed import parallel_state as ps

        with (
            patch.object(ps, "_ENABLE_FLASHINFER_ALLREDUCE_ONLY", True),
            patch.object(ps, "_ATTN_TP", attn_tp),
            patch.object(ps, "_MOE_EP", moe_ep),
            patch.object(ps, "_MOE_TP", moe_tp),
        ):
            ps._tag_groups_for_flashinfer_allreduce_only()

    def test_hybrid_ep_tp_tags_only_the_ep_group(self):
        attn_tp = _FakeGroupCoordinator(4)
        moe_ep = _FakeGroupCoordinator(2)
        moe_tp = _FakeGroupCoordinator(2)

        self._tag(attn_tp=attn_tp, moe_ep=moe_ep, moe_tp=moe_tp)

        self.assertEqual(attn_tp._fi_workspace_hint, "attn_tp")
        self.assertEqual(moe_ep._fi_workspace_hint, "moe")
        self.assertIsNone(moe_tp._fi_workspace_hint)

    def test_pure_moe_tp_tags_only_the_moe_tp_group(self):
        attn_tp = _FakeGroupCoordinator(4)
        moe_ep = _FakeGroupCoordinator(1)
        moe_tp = _FakeGroupCoordinator(4)

        self._tag(attn_tp=attn_tp, moe_ep=moe_ep, moe_tp=moe_tp)

        self.assertEqual(moe_tp._fi_workspace_hint, "moe")
        self.assertIsNone(moe_ep._fi_workspace_hint)

    def test_shared_coordinator_prefers_attn_tp(self):
        # tp=4, ep=4: _ATTN_TP is _MOE_EP is _TP. Either workspace spans the
        # same peers, but the choice must be deterministic.
        shared = _FakeGroupCoordinator(4)
        moe_tp = _FakeGroupCoordinator(1)

        self._tag(attn_tp=shared, moe_ep=shared, moe_tp=moe_tp)

        self.assertEqual(shared._fi_workspace_hint, "attn_tp")
        self.assertIsNone(moe_tp._fi_workspace_hint)


if __name__ == "__main__":
    unittest.main()
