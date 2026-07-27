"""`from_layer` and the serving wrapper against a genuine FusedMoE (plan
section 31.6, W4.2).

The runner-attach suite validates admission against a FAKED base layer, and
the pipeline suite constructs the runner with explicitly-known parameters.
Neither catches the failure modes this file guards:

* `from_layer`'s EXTRACTION drifting — a `moe_runner_config` field rename, a
  weight-shape indexing change (`w2_weight.shape[2]` vs `[1]`), a dispatcher
  default flip — while both other suites keep passing, because neither reads
  those values off a real layer.
* the production wrapper (`FusedMoEWithLoRA._forward_sgl_lora` plus its
  `_get_sgl_lora_batch` view) drifting from the engine it drives: every other
  suite calls `SglMoeLoraRunner.run` directly and never executes the
  layer-level forward that serving actually dispatches.
"""

import sys
import unittest
from pathlib import Path

import msgspec
import torch

from sglang.srt.distributed import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.lora.utils import LoRABatchInfo, MoELoRABatchInfo
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

# The serial control and the signal-gate engine live in the benchmark
# laboratory (one implementation for CI and lab); anchor the import at the
# repo root, exactly as test_sgl_lora_moe_pipeline.py does.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.kernels.lora_moe.cases import (  # noqa: E402
    AdapterCell,
    build_case,
    materialize_case_tensors,
)
from benchmark.kernels.lora_moe.reference import reference_local_moe  # noqa: E402
from benchmark.kernels.lora_moe.serial_control import (  # noqa: E402
    run_base_only_torch,
    run_serial_materialized_control,
)
from benchmark.kernels.lora_moe.signal_gates import (  # noqa: E402
    check_delta,
    require_bitwise_equal,
    resolve_signal_gates,
)


class _StubLoraBackend(msgspec.Struct, kw_only=True):
    """The exact backend surface FusedMoEWithLoRA touches on the sgl_lora path.

    `FusedMoEWithLoRA.__init__` writes ``is_moe_lora``; `_get_sgl_lora_batch`
    reads ``batch_info.moe_lora_info``.  A ``msgspec.Struct`` has no
    ``__dict__``, so any NEW backend attribute the wrapper starts touching
    fails loudly here instead of silently passing against a mock.
    """

    batch_info: LoRABatchInfo
    is_moe_lora: bool = False


def _stub_backend(
    *, token_slots: torch.Tensor, adapter_enabled: torch.Tensor
) -> _StubLoraBackend:
    """Backend stand-in carrying the REAL batch-info containers.

    Only ``moe_lora_info.token_lora_mapping`` / ``.adapter_enabled`` are read
    on the sgl_lora path; every other ``LoRABatchInfo`` field is an inert
    constructor-satisfying placeholder, present so a field rename in the real
    containers breaks this construction rather than being hidden by a mock.
    """
    num_tokens = token_slots.shape[0]
    device = token_slots.device
    placeholder = torch.zeros(1, dtype=torch.int32, device=device)
    moe_lora_info = MoELoRABatchInfo(
        seg_indptr=torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        req_to_lora=placeholder,
        adapter_enabled=adapter_enabled,
        token_lora_mapping=token_slots,
    )
    batch_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=moe_lora_info.seg_indptr,
        weight_indices=placeholder,
        lora_ranks=placeholder,
        scalings=placeholder,
        max_len=num_tokens,
        seg_lens=None,
        permutation=None,
        moe_lora_info=moe_lora_info,
    )
    return _StubLoraBackend(batch_info=batch_info)


class TestSglLoraFromRealLayer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("masked BF16 DeepGEMM requires SM90+")
        try:
            import deep_gemm  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("DeepGEMM required") from error
        torch.cuda.set_device(0)
        # A real FusedMoE reads global server args at construction (runtime
        # context); the deep_gemm backend makes admission resolve without a
        # patched flag.
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", moe_runner_backend="deep_gemm")
        )
        # ServerArgs resolution keeps unquantized layers on TRITON; the layer
        # must resolve DEEP_GEMM at construction for admission to accept it,
        # so pin the runtime flag directly (and restore it in tearDownClass).
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.runtime_context import get_flags

        cls._saved_backend = get_flags().moe.runner_backend
        get_flags().moe.runner_backend = MoeRunnerBackend.DEEP_GEMM
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1, expert_model_parallel_size=1
        )
        cls.device = torch.device("cuda:0")

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.runtime_context import get_flags

        get_flags().moe.runner_backend = cls._saved_backend
        destroy_model_parallel()

    def setUp(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.lora.sgl_lora.moe_lora_runner import SglMoeLoraBatch

        self.num_experts, self.hidden, self.intermediate = 8, 128, 128
        self.top_k, self.num_tokens, self.rank, self.num_adapters = 2, 12, 8, 2
        self.routed_scale = 1.75
        generator = torch.Generator(device="cpu").manual_seed(31)

        def rand(*shape, scale=1.0):
            return (
                torch.randn(*shape, generator=generator, dtype=torch.bfloat16) * scale
            ).to(self.device)

        self.layer = FusedMoE(
            num_experts=self.num_experts,
            hidden_size=self.hidden,
            intermediate_size=self.intermediate,
            layer_id=0,
            top_k=self.top_k,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            gate_up_interleaved=False,
            routed_scaling_factor=self.routed_scale,
        ).to(self.device)
        with torch.no_grad():
            self.layer.w13_weight.copy_(
                rand(self.num_experts, 2 * self.intermediate, self.hidden, scale=0.05)
            )
            self.layer.w2_weight.copy_(
                rand(self.num_experts, self.hidden, self.intermediate, scale=0.05)
            )
        self.batch = SglMoeLoraBatch(
            gate_up_lora_a=rand(
                self.num_adapters,
                self.num_experts,
                2 * self.rank,
                self.hidden,
                scale=0.1,
            ),
            gate_up_lora_b=rand(
                self.num_adapters,
                self.num_experts,
                2 * self.intermediate,
                self.rank,
                scale=0.1,
            ),
            down_lora_a=rand(
                self.num_adapters,
                self.num_experts,
                self.rank,
                self.intermediate,
                scale=0.1,
            ),
            down_lora_b=rand(
                self.num_adapters, self.num_experts, self.hidden, self.rank, scale=0.1
            ),
            token_slots=torch.tensor(
                [0, 1, -1, 0] * (self.num_tokens // 4),
                dtype=torch.int32,
                device=self.device,
            ),
            adapter_enabled=None,
            physical_rank=self.rank,
            shared_outer=False,
        )
        self.hidden_states = rand(self.num_tokens, self.hidden)
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, generator=generator), dim=-1
        ).to(self.device)
        topk_ids = torch.stack(
            [
                torch.randperm(self.num_experts, generator=generator)[: self.top_k]
                for _ in range(self.num_tokens)
            ]
        ).to(device=self.device, dtype=torch.int32)
        self.topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        )

    def _manual_forward(self, *, runner, batch, output_dtype=None):
        """The engine path the wrapper must reproduce: dispatch -> run -> combine."""
        dispatch_output = self.layer.dispatcher.dispatch(
            self.hidden_states.clone(), self.topk_output
        )
        combine_input = runner.run(dispatch_output, batch, output_dtype=output_dtype)
        return self.layer.dispatcher.combine(combine_input)

    def test_from_layer_runner_matches_explicit_construction_bitwise(self):
        """Admission accepts a real layer; extraction produces the same runner.

        Both runners share the SAME weight and factor tensors, so any output
        difference can only come from `from_layer`'s reads off the layer
        (top_k, routed scaling, expert count, w2-derived geometry, the
        provider's layout flags). Bitwise equality is the correct gate: the
        kernels are deterministic, so extraction drift is the only possible
        source of divergence.
        """
        from sglang.srt.environ import envs
        from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )
        from sglang.srt.lora.sgl_lora.moe_lora_runner import SglMoeLoraRunner
        from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

        from_layer_runner = SglMoeLoraRunner.from_layer(self.layer)
        # The provider selector is enumerable: unknown names fail at attach,
        # and the cutedsl provider constructs on this SM100+ device. Both
        # branches guard the selection logic no other test touches.
        with envs.SGLANG_LORA_MOE_BASE_PROVIDER.override("nonsense"):
            with self.assertRaisesRegex(ValueError, "nonsense"):
                SglMoeLoraRunner.from_layer(self.layer)
        if torch.cuda.get_device_capability() >= (9, 0):
            with envs.SGLANG_LORA_MOE_BASE_PROVIDER.override("cutedsl"):
                cutedsl_runner = SglMoeLoraRunner.from_layer(self.layer)
            self.assertEqual(cutedsl_runner.provider.contract.key, "cutedsl_bf16")

        self.assertEqual(from_layer_runner.top_k, self.top_k)
        self.assertEqual(from_layer_runner.provider.num_local_experts, self.num_experts)
        self.assertEqual(from_layer_runner.routed_scaling_factor, self.routed_scale)

        explicit_runner = SglMoeLoraRunner(
            provider=DeepGemmBf16Provider(
                SglLoraBf16QuantInfo(
                    w13_weight=self.layer.w13_weight,
                    w2_weight=self.layer.w2_weight,
                    num_local_experts=self.num_experts,
                    intermediate_size=self.intermediate,
                    hidden_size=self.hidden,
                )
            ),
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scale,
        )

        outputs = {
            name: self._manual_forward(runner=runner, batch=self.batch)
            for name, runner in (
                ("from_layer", from_layer_runner),
                ("explicit", explicit_runner),
            )
        }
        self.assertTrue(
            torch.equal(outputs["from_layer"], outputs["explicit"]),
            "from_layer extraction diverged from explicit construction",
        )
        self.assertTrue(torch.isfinite(outputs["from_layer"]).all())
        # LoRA must have signal in this construction, or the equality above
        # would vacuously pass on the base path alone.
        base_batch = msgspec.structs.replace(
            self.batch, token_slots=torch.full_like(self.batch.token_slots, -1)
        )
        base_out = self._manual_forward(runner=from_layer_runner, batch=base_batch)
        self.assertFalse(
            torch.equal(outputs["from_layer"], base_out),
            "case has no LoRA signal; the bitwise gate would be vacuous",
        )

    def test_wrapper_forward_matches_manual_engine_path_bitwise(self):
        """The serving wrapper is pure plumbing over the engine and must stay so.

        `forward` dispatches on ``lora_execution_engine`` to
        `_forward_sgl_lora`, which builds the `SglMoeLoraBatch` view straight
        from ``batch_info.moe_lora_info`` plus the factors bound by
        `set_lora_info`, and hands ``output_dtype`` through to the runner.
        Red on: batch-view wiring drift (wrong mapping field, physical_rank no
        longer ``down_lora_a.shape[2]``, dropped adapter_enabled/shared_outer),
        forward falling back to the legacy engine, or the ``output_dtype``
        kwarg being dropped (the FP32 arm's dtype assert catches that even if
        values round-trip). Bitwise is the correct gate: both sides drive the
        same deterministic kernels over the same tensors, so ANY divergence is
        plumbing drift. The fixture case carries LoRA signal (asserted by the
        from_layer test), so a wrapper that degrades to base-only cannot pass.
        """
        from sglang.srt.lora.layers import FusedMoEWithLoRA, get_lora_layer
        from sglang.srt.lora.sgl_lora.moe_lora_runner import SglMoeLoraRunner

        adapter_enabled = torch.ones(
            self.num_adapters, dtype=torch.int32, device=self.device
        )
        stub = _stub_backend(
            token_slots=self.batch.token_slots, adapter_enabled=adapter_enabled
        )
        wrapper = get_lora_layer(
            layer=self.layer, lora_backend=stub, lora_execution_engine="sgl_lora"
        )
        self.assertIsInstance(wrapper, FusedMoEWithLoRA)
        # The wrapper must mark the backend: serving batch prep only builds
        # moe_lora_info when is_moe_lora is set.
        self.assertTrue(stub.is_moe_lora)
        wrapper.set_lora_info(
            gate_up_lora_a_weights=self.batch.gate_up_lora_a,
            gate_up_lora_b_weights=self.batch.gate_up_lora_b,
            down_lora_a_weights=self.batch.down_lora_a,
            down_lora_b_weights=self.batch.down_lora_b,
        )

        manual_runner = SglMoeLoraRunner.from_layer(self.layer)
        manual_batch = msgspec.structs.replace(
            self.batch, adapter_enabled=adapter_enabled
        )
        for output_dtype in (None, torch.float32):
            with self.subTest(output_dtype=output_dtype):
                manual = self._manual_forward(
                    runner=manual_runner, batch=manual_batch, output_dtype=output_dtype
                )
                kwargs = {} if output_dtype is None else {"output_dtype": output_dtype}
                observed = wrapper(
                    hidden_states=self.hidden_states.clone(),
                    topk_output=self.topk_output,
                    **kwargs,
                )
                self.assertEqual(observed.dtype, manual.dtype)
                self.assertTrue(
                    torch.equal(observed, manual),
                    "wrapper forward diverged from the manual "
                    "dispatch -> run -> combine path",
                )

    def test_wrapper_forward_matches_serial_control(self):
        """W4.2's closing gate: forward THROUGH the wrapper vs the lab control.

        A real FusedMoE, wrapped exactly as serving wraps it, forwarded
        through `_forward_sgl_lora`, adjudicated against the lab's independent
        `serial_materialized_control` on the SAME case tensors under the same
        signal gates as the guardrail matrix (check_step1_correctness). Red
        on: any wrapper-or-engine change that shifts the layer-level output
        beyond the matched-base gates — including wrapper and engine moving
        TOGETHER, which the bitwise test above cannot see.

        Comparison discipline mirrors the matrix: each side subtracts its OWN
        matched all-base forward so BF16 base noise cancels and a large base
        cannot hide a wrong LoRA delta.
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.lora.layers import get_lora_layer

        case = build_case(
            device="cuda",
            model_preset="tiny_smoke",
            adapter_cell=AdapterCell(
                active_adapters=2, include_base_rows=True, slot_capacity=4
            ),
            route_generator="iid",
            num_tokens=16,
            active_rank=16,
            routed_scaling_factor=1.75,
            seed=11,
            # Explicit identity keeps the case deterministic and avoids the
            # git subprocess of capture_source_revision in CI.
            source_revision="registered-test",
        )
        tensors = materialize_case_tensors(case)
        control = run_serial_materialized_control(case, tensors, device=self.device)
        control_base = run_base_only_torch(case, tensors, device=self.device)

        layer = FusedMoE(
            num_experts=case.num_experts_local,
            hidden_size=case.moe_hidden_size,
            intermediate_size=case.intermediate_size_local,
            layer_id=0,
            top_k=case.top_k,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            gate_up_interleaved=False,
            routed_scaling_factor=case.routed_scaling_factor,
        ).to(self.device)
        with torch.no_grad():
            layer.w13_weight.copy_(tensors.w13)
            layer.w2_weight.copy_(tensors.w2)

        # int32 mapping is what production _compute_moe_lora_info emits.
        token_slots = tensors.token_lora_mapping.to(
            device=self.device, dtype=torch.int32
        )
        stub = _stub_backend(
            token_slots=token_slots,
            adapter_enabled=torch.ones(
                case.slot_capacity, dtype=torch.int32, device=self.device
            ),
        )
        wrapper = get_lora_layer(
            layer=layer, lora_backend=stub, lora_execution_engine="sgl_lora"
        )
        wrapper.set_lora_info(
            gate_up_lora_a_weights=tensors.lora_a_gate_up.to(self.device),
            gate_up_lora_b_weights=tensors.lora_b_gate_up.to(self.device),
            down_lora_a_weights=tensors.lora_a_down.to(self.device),
            down_lora_b_weights=tensors.lora_b_down.to(self.device),
        )
        hidden_states = tensors.hidden_states.to(self.device)
        topk_output = StandardTopKOutput(
            topk_weights=tensors.topk_weights.to(self.device),
            topk_ids=tensors.topk_ids.to(self.device),
            router_logits=None,
        )
        wrapper_out = wrapper(
            hidden_states=hidden_states.clone(), topk_output=topk_output
        )
        self.assertEqual(wrapper_out.dtype, torch.bfloat16)
        # The wrapper's matched base: the same forward with every token on the
        # -1 sentinel, exactly how production expresses a base-only batch.
        stub.batch_info.moe_lora_info.token_lora_mapping = torch.full_like(
            token_slots, -1
        )
        wrapper_base = wrapper(
            hidden_states=hidden_states.clone(), topk_output=topk_output
        )

        reference_base = reference_local_moe(case, tensors, include_lora=False)
        reference_full = reference_local_moe(case, tensors)
        gates = resolve_signal_gates(
            reference_full - reference_base,
            gate_dtype=torch.bfloat16,
            base_reference=reference_base,
        )
        for label, observed, observed_base in (
            ("serial control", control.output, control_base),
            ("wrapper forward", wrapper_out, wrapper_base),
        ):
            record = check_delta(
                observed.cpu().float() - observed_base.cpu().float(),
                reference_full - reference_base,
                gates,
                label=f"{label} e2e",
            )
            self.assertTrue(record.passed, msg=str(record))
        # Mixed traffic through the wrapper: base rows must ride the LoRA
        # topology untouched (assert the seed actually produced base rows, or
        # this check is vacuous).
        self.assertTrue(case.observed_base_rows)
        base_rows = (tensors.token_lora_mapping == -1).to(self.device)
        require_bitwise_equal(
            wrapper_out[base_rows],
            wrapper_base[base_rows],
            label="wrapper mixed base rows bitwise",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
