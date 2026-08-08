"""End-to-end test for the SGL LoRA MoE runner on its own base-GEMM provider.

Exercises the full own-stack pipeline (`SglMoeLoraRunner.run` over
`DeepGemmBf16BaseGemm`) against an independent staged reference for base-only,
mixed, and active adapter traffic in both destination dtypes.  The LoRA
contribution is compared after subtracting each side's own matched base output
so a large base cannot hide a wrong delta.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
    DeepGemmBf16Provider,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import (
    PROVISIONAL_LAUNCH_CONFIG,
    SglMoeLoraBatch,
    SglMoeLoraRunner,
)
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=140, stage="base-b", runner_config="1-gpu-small")

# The signal-gate engine lives in the benchmark laboratory (one implementation
# for CI and lab); anchor the import at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.kernels.lora_moe.signal_gates import (  # noqa: E402
    check_delta,
    require_bitwise_equal,
    require_delta_close,
    resolve_signal_gates,
)


class TestSglLoraMoePipeline(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest(
                "masked BF16 DeepGEMM requires an SM90-or-newer NVIDIA GPU"
            )
        try:
            import deep_gemm  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(
                "the DeepGEMM Python package is required"
            ) from error

        torch.cuda.set_device(0)
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def _bf16_linear(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input_.float(), weight.float().T).to(torch.bfloat16)

    def _reference(
        self,
        token_lora_mapping: torch.Tensor,
        output_dtype: torch.dtype,
        *,
        include_lora: bool,
    ) -> torch.Tensor:
        """Staged BF16 reference with a fixed-order FP32 top-k reduction."""
        pair_outputs = []
        for token_id in range(self.hidden_states.shape[0]):
            adapter_id = int(token_lora_mapping[token_id].item())
            if not include_lora:
                adapter_id = -1
            token_outputs = []
            for slot in range(self.topk):
                expert_id = int(self.topk_ids[token_id, slot].item())
                gate_up = self._bf16_linear(
                    self.hidden_states[token_id], self.w13_weight[expert_id]
                )
                if adapter_id >= 0:
                    gate_a = self._bf16_linear(
                        self.hidden_states[token_id],
                        self.gate_a[adapter_id, expert_id],
                    )
                    gate_delta = torch.cat(
                        (
                            self._bf16_linear(
                                gate_a[: self.rank],
                                self.gate_b[adapter_id, expert_id, : self.intermediate],
                            ),
                            self._bf16_linear(
                                gate_a[self.rank :],
                                self.gate_b[adapter_id, expert_id, self.intermediate :],
                            ),
                        )
                    )
                    gate_up = (gate_up.float() + gate_delta.float()).to(torch.bfloat16)

                gate = gate_up[: self.intermediate].float()
                up = gate_up[self.intermediate :].float()
                activation = (up * gate * torch.sigmoid(gate)).to(torch.bfloat16)
                down = self._bf16_linear(activation, self.w2_weight[expert_id])
                if adapter_id >= 0:
                    down_a = self._bf16_linear(
                        activation, self.down_a[adapter_id, expert_id]
                    )
                    down_delta = self._bf16_linear(
                        down_a, self.down_b[adapter_id, expert_id]
                    )
                    down = (down.float() + down_delta.float()).to(torch.bfloat16)
                token_outputs.append(down)
            pair_outputs.append(torch.stack(token_outputs))

        pairs = torch.stack(pair_outputs)
        combined = torch.zeros(
            self.hidden_states.shape, dtype=torch.float32, device=self.device
        )
        for slot in range(self.topk):
            combined = (
                combined + pairs[:, slot].float() * self.topk_weights[:, slot, None]
            )
        return (combined * self.routed_scale).to(output_dtype)

    def setUp(self):
        torch.manual_seed(47)
        self.device = torch.device("cuda")
        self.num_experts = 2
        self.num_adapters = 2
        self.hidden_size = 128
        self.intermediate = 128
        self.rank = 16
        self.topk = 2
        self.routed_scale = 1.75

        def scaled(*shape: int, scale: float) -> torch.Tensor:
            return (torch.randn(shape, device=self.device) * scale).to(torch.bfloat16)

        self.hidden_states = scaled(4, self.hidden_size, scale=0.2)
        self.topk_ids = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int32, device=self.device
        )
        self.topk_weights = torch.tensor(
            [[0.7, 0.3], [0.2, 0.8], [0.55, 0.45], [0.35, 0.65]],
            dtype=torch.float32,
            device=self.device,
        )
        self.w13_weight = scaled(
            self.num_experts, 2 * self.intermediate, self.hidden_size, scale=0.1
        )
        self.w2_weight = scaled(
            self.num_experts, self.hidden_size, self.intermediate, scale=0.1
        )
        self.gate_a = scaled(
            self.num_adapters,
            self.num_experts,
            2 * self.rank,
            self.hidden_size,
            scale=0.1,
        )
        # B factors clear the signal-gate validity floor (>= 32 BF16 quanta of
        # the base output).
        self.gate_b = scaled(
            self.num_adapters,
            self.num_experts,
            2 * self.intermediate,
            self.rank,
            scale=0.2,
        )
        self.down_a = scaled(
            self.num_adapters,
            self.num_experts,
            self.rank,
            self.intermediate,
            scale=0.1,
        )
        self.down_b = scaled(
            self.num_adapters, self.num_experts, self.hidden_size, self.rank, scale=0.2
        )

        self.config = MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate,
            top_k=self.topk,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            no_combine=False,
            routed_scaling_factor=self.routed_scale,
        )
        self.quant_info = SglLoraBf16QuantInfo(
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
            num_local_experts=self.num_experts,
            intermediate_size=self.intermediate,
            hidden_size=self.hidden_size,
        )
        # Construct the runner directly: `from_layer` needs a full FusedMoE,
        # while this suite exercises the pipeline against a hand-built
        # provider. Serving, lab, and tests share one provisional-tile set.
        self.launch_config = PROVISIONAL_LAUNCH_CONFIG
        self.runner = SglMoeLoraRunner(
            provider=DeepGemmBf16Provider(self.quant_info),
            top_k=self.topk,
            routed_scaling_factor=self.routed_scale,
            launch_config=self.launch_config,
        )
        self.dispatcher = StandardDispatcher(self.config)

    def _run(
        self,
        mapping: tuple[int, ...],
        output_dtype: torch.dtype,
        *,
        adapter_enabled: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        token_lora_mapping = torch.tensor(
            mapping, dtype=torch.int32, device=self.device
        )
        batch = SglMoeLoraBatch(
            gate_up_lora_a=self.gate_a,
            gate_up_lora_b=self.gate_b,
            down_lora_a=self.down_a,
            down_lora_b=self.down_b,
            token_slots=token_lora_mapping,
            adapter_enabled=(
                None
                if adapter_enabled is None
                else torch.tensor(
                    adapter_enabled, dtype=torch.int32, device=self.device
                )
            ),
            physical_rank=self.rank,
            shared_outer=False,
        )
        topk_output = StandardTopKOutput(
            topk_weights=self.topk_weights,
            topk_ids=self.topk_ids,
            router_logits=None,
        )
        dispatch_output = self.dispatcher.dispatch(
            self.hidden_states.clone(), topk_output
        )
        combine_input = self.runner.run(
            dispatch_output, batch, output_dtype=output_dtype
        )
        return self.dispatcher.combine(combine_input)

    def test_base_mixed_and_active_match_staged_reference(self):
        mappings = {
            "base": (-1, -1, -1, -1),
            "mixed": (0, -1, 1, -1),
            "active": (0, 1, 0, 1),
        }
        for output_dtype in (torch.bfloat16, torch.float32):
            outputs = {
                name: self._run(
                    mapping, output_dtype, adapter_enabled=(1,) * self.num_adapters
                )
                for name, mapping in mappings.items()
            }
            reference_base = self._reference(
                torch.tensor(mappings["base"], device=self.device),
                output_dtype,
                include_lora=False,
            )
            with self.subTest(output_dtype=output_dtype, arm="base"):
                # The base arm's own output IS the compared signal.
                require_delta_close(
                    outputs["base"],
                    reference_base,
                    gate_dtype=torch.bfloat16,
                    label=f"base output {output_dtype}",
                )
            for name in ("mixed", "active"):
                with self.subTest(output_dtype=output_dtype, arm=name):
                    reference_full = self._reference(
                        torch.tensor(mappings[name], device=self.device),
                        output_dtype,
                        include_lora=True,
                    )
                    gates = resolve_signal_gates(
                        reference_full.float() - reference_base.float(),
                        gate_dtype=torch.bfloat16,
                        base_reference=reference_base,
                    )
                    record = check_delta(
                        outputs[name].float() - outputs["base"].float(),
                        reference_full.float() - reference_base.float(),
                        gates,
                        label=f"{name} LoRA delta {output_dtype}",
                    )
                    self.assertTrue(record.passed, msg=str(record))

    def test_base_only_assignment_is_bitwise_repeatable(self):
        """A base-only batch rides the LoRA topology and must be stable.

        Guards the unified-graph property: sentinel assignments contribute
        exact zeros, so repeated execution is bitwise identical.
        """
        first = self._run((-1, -1, -1, -1), torch.bfloat16)
        for replay in range(8):
            require_bitwise_equal(
                self._run((-1, -1, -1, -1), torch.bfloat16),
                first,
                label=f"base-only replay {replay}",
            )

    def test_disabled_resident_slot_equals_sentinel_bitwise(self):
        """Serving marks base rows with a real slot plus adapter_enabled=0.

        Regression: routing that ignored ``adapter_enabled`` built routed LoRA
        work for base rows against zero-filled factors — numerically harmless
        but it changed route padding and group counts. Slot 0 disabled must
        route bitwise-identically to the ``-1`` sentinel.
        """
        sentinel = self._run((-1, 1, -1, 1), torch.bfloat16)
        # Slot 0 present in the mapping but disabled; slot 1 stays active.
        disabled_slot = self._run((0, 1, 0, 1), torch.bfloat16, adapter_enabled=(0, 1))
        require_bitwise_equal(
            disabled_slot, sentinel, label="disabled slot vs -1 sentinel"
        )

    def test_rejects_unsupported_output_dtype(self):
        with self.assertRaises(ValueError):
            self._run((0, 1, 0, 1), torch.float16)

    def test_production_tile_selection_xwide_at_m96(self):
        """The PRODUCTION-selected xwide tier, not a forced one (4th review pass).

        The per-stage benchmark forces tile widths and therefore bypasses
        `_token_width_for`; the matrix only reaches expected_m=64. This case
        drives the un-forced selection path at the re-sited threshold: a
        balanced route with exactly 96 rows per expert must SELECT the
        (128, 128) tile, and both stages must stay bitwise-identical to the
        DeepGEMM provider at that shape.
        """
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("the CuTeDSL provider requires SM90+")
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
            CuteDslBf16Provider,
        )
        from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )
        from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

        torch.manual_seed(53)
        num_experts, hidden, intermediate, top_k = 32, 512, 256, 8
        rows = 96  # expected_m lands exactly on the xwide threshold
        num_tokens = num_experts * rows // top_k
        quant_info = SglLoraBf16QuantInfo(
            w13_weight=torch.randn(
                (num_experts, 2 * intermediate, hidden),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.05,
            w2_weight=torch.randn(
                (num_experts, hidden, intermediate),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.05,
            num_local_experts=num_experts,
            intermediate_size=intermediate,
            hidden_size=hidden,
        )
        cute = CuteDslBf16Provider(quant_info)
        deep = DeepGemmBf16Provider(quant_info)

        hidden_states = torch.randn(
            (num_tokens, hidden), dtype=torch.bfloat16, device=self.device
        )
        topk_ids = (
            (torch.arange(num_tokens * top_k, dtype=torch.int32) % num_experts)
            .view(num_tokens, top_k)
            .to(self.device)
        )
        ws = cute.prepare(hidden_states, topk_ids, top_k)
        self.assertEqual(int(ws.expected_m), rows)
        # The [:, :rows] comparisons below assume UNIFORM occupancy; assert it
        # so a route-construction change cannot silently narrow the check.
        self.assertTrue(bool((ws.masked_m == rows).all()))
        # Pin the selection on both sides of the re-sited threshold.
        self.assertEqual(cute._token_width_for(ws.m_max, rows), 128)
        self.assertEqual(cute._token_width_for(ws.m_max, rows - 1), 64)
        # Packing escalation (review regression): expected_m=64 prefers the
        # wide tile, but m_max=65792 exceeds its 64x1024 packing, so the
        # selector must escalate to the compiled xwide tile instead of
        # raising (reachable at E=1024/top_k=1/T=65536).
        self.assertEqual(cute._token_width_for(65792, 64), 128)

        # BOTH engines consume the SAME workspace (the masked-row-domain
        # refactor makes this legal: S2/S4 read only the base fields), which
        # is the only sound comparison: the S1 preprocess assigns intra-expert
        # row order via atomics, so two separate prepare() calls permute rows
        # differently and raw-buffer comparisons across them are meaningless.
        # Compare VALID rows only — rows beyond masked_m are padding no engine
        # contracts to write.
        providers = (("cute", cute), ("deep", deep))
        gateup_outs = {}
        for name, provider in providers:
            out = torch.empty(
                provider.gateup_out_shape(ws),
                dtype=torch.bfloat16,
                device=self.device,
            )
            provider.gateup(ws, out)
            gateup_outs[name] = out[:, :rows]
        self.assertTrue(
            torch.equal(gateup_outs["cute"], gateup_outs["deep"]),
            "gateup diverged across providers at the xwide-selected shape",
        )

        act = (
            torch.randn(
                cute.act_out_shape(ws), dtype=torch.bfloat16, device=self.device
            )
            * 0.05
        )
        down_outs = {}
        for name, provider in providers:
            out = torch.empty(
                provider.down_out_shape(ws),
                dtype=torch.bfloat16,
                device=self.device,
            )
            provider.down(ws, act, out)
            down_outs[name] = out[:, :rows]
        self.assertTrue(
            torch.equal(down_outs["cute"], down_outs["deep"]),
            "down diverged across providers at the xwide-selected shape",
        )

    def test_shared_compile_cache_keeps_per_layer_weights(self):
        """A cached compiled function must still read ITS OWN layer's weights.

        The provider shares one compiled CuTeDSL function process-wide (a
        60-layer model otherwise pays ~2.8 min of compile per server start).
        That is only sound because the resident weight is a runtime argument;
        if a future change cached the weight WRAPPER alongside the function,
        every layer would silently compute with the first layer's weights —
        wrong output, no error. Red on exactly that: two providers built from
        different weights must each match DeepGEMM on their own weights, and
        must disagree with each other.
        """
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("the CuTeDSL provider requires SM90+")
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
            CuteDslBf16Provider,
        )
        from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )
        from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

        num_experts, hidden, intermediate, top_k = 8, 256, 128, 4
        rows = 32
        num_tokens = num_experts * rows // top_k

        def layer(seed: int) -> SglLoraBf16QuantInfo:
            gen = torch.Generator(device="cpu").manual_seed(seed)

            def rand(*shape):
                return (torch.randn(*shape, generator=gen) * 0.05).to(
                    device=self.device, dtype=torch.bfloat16
                )

            return SglLoraBf16QuantInfo(
                w13_weight=rand(num_experts, 2 * intermediate, hidden),
                w2_weight=rand(num_experts, hidden, intermediate),
                num_local_experts=num_experts,
                intermediate_size=intermediate,
                hidden_size=hidden,
            )

        torch.manual_seed(59)
        hidden_states = (torch.randn(num_tokens, hidden, device=self.device) * 0.2).to(
            torch.bfloat16
        )
        topk_ids = (
            (torch.arange(num_tokens * top_k, dtype=torch.int32) % num_experts)
            .view(num_tokens, top_k)
            .to(self.device)
        )

        # BOTH stages: the cache shares six compiled functions (3 tiles x
        # gemm1/gemm2), and retaining the first layer's W2 wrapper would
        # corrupt every later layer's down() while a gateup-only test stays
        # green (review finding).
        gateup_outputs, down_outputs = [], []
        act = (
            torch.randn(
                num_experts,
                256,
                intermediate,
                device=self.device,
            )
            * 0.05
        ).to(torch.bfloat16)
        for seed in (101, 202):
            info = layer(seed)
            cute, deep = CuteDslBf16Provider(info), DeepGemmBf16Provider(info)
            ws = cute.prepare(hidden_states, topk_ids, top_k)
            self.assertTrue(bool((ws.masked_m == rows).all()))
            # One FIXED activation for both layers, so a down() difference can
            # only come from W2 — a fresh random act per layer would make the
            # cross-layer inequality vacuous.
            self.assertLessEqual(ws.m_max, act.shape[1])
            per_stage = {}
            for stage in ("gateup", "down"):
                per_provider = {}
                for name, provider in (("cute", cute), ("deep", deep)):
                    if stage == "gateup":
                        out = torch.empty(
                            provider.gateup_out_shape(ws),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        provider.gateup(ws, out)
                    else:
                        out = torch.empty(
                            provider.down_out_shape(ws),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        provider.down(ws, act[:, : ws.m_max], out)
                    per_provider[name] = out[:, :rows]
                self.assertTrue(
                    torch.equal(per_provider["cute"], per_provider["deep"]),
                    f"the shared compiled {stage} function used the wrong "
                    "layer's weights",
                )
                per_stage[stage] = per_provider["cute"].clone()
            gateup_outputs.append(per_stage["gateup"])
            down_outputs.append(per_stage["down"])

        for label, outputs in (("gateup", gateup_outputs), ("down", down_outputs)):
            self.assertFalse(
                torch.equal(outputs[0], outputs[1]),
                f"two layers with different weights produced identical {label} "
                "output; the shared compiled function is reading one layer's "
                "weights",
            )

    def test_cutedsl_rejects_expert_counts_beyond_the_packing_at_attach(self):
        """E > 1024 must fail at ATTACH, not on the first forward.

        The direct-schedule ABI packs expert indices in 10 bits; without this
        gate the builder's guard fires per-forward, after admission already
        accepted the layer (review follow-up).
        """
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("the CuTeDSL provider requires SM90+")
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
            CuteDslBf16Provider,
        )
        from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

        experts, hidden, intermediate = 1025, 64, 32
        info = SglLoraBf16QuantInfo(
            w13_weight=torch.zeros(
                (experts, 2 * intermediate, hidden),
                dtype=torch.bfloat16,
                device=self.device,
            ),
            w2_weight=torch.zeros(
                (experts, hidden, intermediate),
                dtype=torch.bfloat16,
                device=self.device,
            ),
            num_local_experts=experts,
            intermediate_size=intermediate,
            hidden_size=hidden,
        )
        with self.assertRaisesRegex(ValueError, "expert"):
            CuteDslBf16Provider(info)

    def test_provider_graph_replay_matches_eager_after_mutation(self):
        """Capture -> mutate inputs -> replay must equal a fresh eager run.

        The timing harness proved graph LAUNCHABILITY but discarded outputs,
        so nothing observed graph CORRECTNESS. The first version of this test
        had its own observability hole (review finding): gateup's output was
        discarded and down consumed a fixed pre-capture activation, so
        mutating hidden_states could not affect any asserted value — a stale
        replayed gate/up GEMM passed. This version captures the COMPLETE
        chain prepare -> gateup -> activation -> down -> finalize and asserts
        the final TOKEN-ORDER output, which is also the only
        permutation-safe comparison: the S1 dispatch assigns intra-expert row
        order via atomics, so raw expert rows from two prepare() calls are
        not comparable, while finalize maps every pair home.
        """
        import torch.nn.functional as F

        from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )
        from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

        providers = [("deep", DeepGemmBf16Provider)]
        if torch.cuda.get_device_capability() >= (9, 0):
            from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16Provider,
            )

            providers.append(("cute", CuteDslBf16Provider))

        torch.manual_seed(61)
        num_experts, hidden, intermediate, top_k = 8, 256, 128, 4
        num_tokens = 64
        info = SglLoraBf16QuantInfo(
            w13_weight=(
                torch.randn(num_experts, 2 * intermediate, hidden, device=self.device)
                * 0.05
            ).to(torch.bfloat16),
            w2_weight=(
                torch.randn(num_experts, hidden, intermediate, device=self.device)
                * 0.05
            ).to(torch.bfloat16),
            num_local_experts=num_experts,
            intermediate_size=intermediate,
            hidden_size=hidden,
        )

        def fresh_routing():
            ids = torch.randint(
                0,
                num_experts,
                (num_tokens, top_k),
                device=self.device,
                dtype=torch.int32,
            )
            weights = torch.softmax(
                torch.randn(num_tokens, top_k, device=self.device), dim=-1
            )
            return ids, weights

        for name, provider_cls in providers:
            with self.subTest(provider=name):
                provider = provider_cls(info)
                # STATIC input buffers: the graph reads these addresses, so
                # in-place mutation before replay is the serving pattern.
                hidden_states = (
                    torch.randn(num_tokens, hidden, device=self.device) * 0.2
                ).to(torch.bfloat16)
                topk_ids, topk_weights = fresh_routing()

                def forward():
                    ws = provider.prepare(hidden_states, topk_ids, top_k)
                    gateup = torch.empty(
                        provider.gateup_out_shape(ws),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    provider.gateup(ws, gateup)
                    # The provider contract is gate-first, non-interleaved.
                    gate = gateup[:, :, :intermediate].float()
                    up = gateup[:, :, intermediate:].float()
                    act = (F.silu(gate) * up).to(torch.bfloat16).contiguous()
                    down = torch.empty(
                        provider.down_out_shape(ws),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    provider.down(ws, act, down)
                    output = torch.empty(
                        num_tokens,
                        hidden,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    provider.finalize(
                        ws,
                        down,
                        topk_ids,
                        topk_weights,
                        None,
                        output,
                    )
                    return output

                # Warm outside capture (module load must never happen inside),
                # then capture one full forward.
                forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_output = forward()

                # Mutate the activations AND the routing in place: a replay
                # that latched any capture-time value now diverges in the
                # asserted token-order output.
                hidden_states.copy_(
                    (torch.randn(num_tokens, hidden, device=self.device) * 0.2).to(
                        torch.bfloat16
                    )
                )
                new_ids, new_weights = fresh_routing()
                topk_ids.copy_(new_ids)
                topk_weights.copy_(new_weights)
                graph.replay()
                torch.cuda.synchronize()
                replayed = captured_output.clone()

                eager = forward()
                self.assertFalse(torch.isnan(replayed).any())
                self.assertTrue(
                    torch.equal(replayed, eager),
                    "graph replay diverged from eager on the mutated inputs "
                    f"(max abs diff "
                    f"{(replayed.float() - eager.float()).abs().max().item()})",
                )


if __name__ == "__main__":
    unittest.main()
