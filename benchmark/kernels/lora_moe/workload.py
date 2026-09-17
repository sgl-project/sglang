"""Synthetic resident MoE fixture using the production runner and providers."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version


class Workload:
    def __init__(self, case, seed):
        import torch

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.lora.moe.activation import ActivationFn
        from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
        from sglang.srt.lora.moe.plan import resolve_plans
        from sglang.srt.lora.moe.quant_info import (
            MoeLoraBf16QuantInfo,
            MoeLoraFp8QuantInfo,
        )
        from sglang.srt.lora.moe.runner import MoeLoraBatch
        from sglang.srt.lora.utils import Phase, architecture_for_capability

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda")
        capability = torch.cuda.get_device_capability(device)
        if capability[0] not in (9, 10):
            raise ValueError("this tuner admits SM90/SM100 only")
        self.case, self.device = case, device
        self.phase = Phase.PREFILL if case.phase == "prefill" else Phase.DECODE
        self.selected = resolve_plans(
            architecture=architecture_for_capability(capability[0]),
            is_shared_outer=case.layout == "shared",
            physical_rank=case.rank,
            activation=ActivationFn.SILU,
            hidden_size=case.hidden_size,
            num_local_experts=case.num_local_experts,
            quant_family=case.quant,
        )[self.phase]
        self.incumbent = asdict(self.selected.tiles.config_for(case.tokens))
        g = torch.Generator(device=device).manual_seed(seed)

        def rand(shape, scale):
            return (torch.randn(shape, generator=g, device=device) * scale).to(
                torch.bfloat16
            )

        e, h, i, r, t, slot_count = (
            case.num_local_experts,
            case.hidden_size,
            case.intermediate_size,
            case.rank,
            case.tokens,
            case.slots,
        )
        outer = 1 if case.layout == "shared" else e
        self.weights = {
            "w13": rand((e, 2 * i, h), h**-0.5),
            "w2": rand((e, h, i), i**-0.5),
            "a13": rand((slot_count, outer, 2 * r, h), h**-0.5),
            "b13": rand((slot_count, e, 2 * i, r), r**-0.5),
            "a2": rand((slot_count, e, r, i), i**-0.5),
            "b2": rand((slot_count, outer, h, r), r**-0.5),
        }
        self.hidden = rand((t, h), 0.2)
        ids = (
            torch.arange(t, device=device)[:, None]
            + torch.arange(case.top_k, device=device)[None, :]
        ) % e
        if case.routing == "skewed":
            ids = torch.arange(case.top_k, device=device)[None, :].expand(t, -1)
        self.ids = ids.to(torch.int32).contiguous()
        self.router_weights = torch.softmax(
            torch.randn((t, case.top_k), generator=g, device=device), dim=1
        )
        slots = torch.arange(t, device=device, dtype=torch.int32) % slot_count
        if case.traffic == "mixed":
            slots[torch.arange(t, device=device) % 3 == 0] = -1
        elif case.traffic == "base_only":
            slots.fill_(-1)
        self.slots = slots
        info = dict(
            w13_weight=self.weights["w13"],
            w2_weight=self.weights["w2"],
            num_local_experts=e,
            intermediate_size=i,
            hidden_size=h,
        )
        if case.quant == "fp8":
            info["w13_weight"], info["w13_scale"], self.weights["w13"] = self._quantize(
                self.weights["w13"]
            )
            info["w2_weight"], info["w2_scale"], self.weights["w2"] = self._quantize(
                self.weights["w2"]
            )
            info["block_shape"] = (128, 128)
            quant_info = MoeLoraFp8QuantInfo(**info)
        else:
            quant_info = MoeLoraBf16QuantInfo(**info)
        provider_cls = select_provider_cls(
            self.selected.base_gemm_rows, case.quant, case.vendor
        )
        self.provider = provider_cls(quant_info)
        self.dispatch = StandardDispatchOutput(
            hidden_states=self.hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                self.router_weights, self.ids, torch.zeros((t, e), device=device)
            ),
        )
        self.batch = MoeLoraBatch(
            gate_up_lora_a=self.weights["a13"],
            gate_up_lora_b=self.weights["b13"],
            down_lora_a=self.weights["a2"],
            down_lora_b=self.weights["b2"],
            token_lora_mapping=slots,
            use_cuda_graph=case.mode == "graph",
            is_prefill=case.phase == "prefill",
        )
        self.reference = self._reference()
        self.identity = {
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(capability),
            "gpu_uuid": str(
                getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": version("triton"),
            "plan": self.selected.name,
            "plan_spec": asdict(self.selected.plan),
            "base_gemm_rows": self.selected.base_gemm_rows,
            "provider": provider_cls.__module__ + "." + provider_cls.__name__,
            "base_config": "shipped lookup or provider heuristic; not tuned",
            "activation": "gated_silu",
            "seed": seed,
            "scope": "resident single-GPU shard, ordinary allocation; no serving allocator, TP/EP communication or model TPS",
        }

    def base_config(self):
        """Read the selected provider state; never replace its lookup or cache."""
        import msgspec

        if self.case.vendor == "triton":
            return {str(k): v for k, v in self.provider._configs.items()}
        table = self.provider._config_table
        return {
            "table": msgspec.to_builtins(table) if table is not None else None,
            "tiles": {
                str(k): msgspec.to_builtins(v)
                for k, v in self.provider._tile_configs.items()
            },
        }

    @staticmethod
    def _quantize(weight):
        import torch

        e, n, k = weight.shape
        blocks = weight.float().reshape(e, n // 128, 128, k // 128, 128)
        scales = (
            blocks.abs().amax(dim=(2, 4)).clamp_min(1e-6)
            / torch.finfo(torch.float8_e4m3fn).max
        )
        quant = (blocks / scales[:, :, None, :, None]).to(torch.float8_e4m3fn)
        effective = (quant.float() * scales[:, :, None, :, None]).reshape(e, n, k)
        return quant.reshape(e, n, k), scales.contiguous(), effective

    def _reference(self):
        """Independent FP32 algebra with exact effective quantized weights.

        FP8 provider activation quantization and BF16 intermediates are not
        imitated; their declared numerical error is checked against this oracle.
        """
        import torch
        import torch.nn.functional as F

        c, w = self.case, self.weights
        output = torch.zeros_like(self.hidden, dtype=torch.float32)
        for expert in range(c.num_local_experts):
            token, column = torch.where(self.ids == expert)
            if token.numel() == 0:
                continue
            x = self.hidden[token].float()
            gate_up = x @ w["w13"][expert].float().T
            outer = 0 if c.layout == "shared" else expert
            for slot in range(c.slots):
                mask = self.slots[token] == slot
                for branch in range(2):
                    a = w["a13"][
                        slot, outer, branch * c.rank : (branch + 1) * c.rank
                    ].float()
                    b = w["b13"][
                        slot,
                        expert,
                        branch * c.intermediate_size : (branch + 1)
                        * c.intermediate_size,
                    ].float()
                    gate_up[
                        mask,
                        branch * c.intermediate_size : (branch + 1)
                        * c.intermediate_size,
                    ] += (x[mask] @ a.T) @ b.T
            gate, up = gate_up.chunk(2, dim=1)
            activation = F.silu(gate) * up
            down = activation @ w["w2"][expert].float().T
            for slot in range(c.slots):
                mask = self.slots[token] == slot
                down[mask] += (activation[mask] @ w["a2"][slot, expert].float().T) @ w[
                    "b2"
                ][slot, outer].float().T
            output.index_add_(0, token, down * self.router_weights[token, column, None])
        return output * c.routed_scaling_factor

    def bind(self, spec):
        import torch

        from sglang.srt.lora.moe.activation import ActivationFn
        from sglang.srt.lora.moe.plan import (
            MoeLoraLaunchConfig,
            SelectedPlan,
            TileTable,
        )
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        class ResidentRunner(MoeLoraRunner):
            def _allocate_output(self, *, num_tokens, dtype, device):
                return torch.empty(
                    (num_tokens, self.hidden_size), dtype=dtype, device=device
                )

        config = MoeLoraLaunchConfig(**spec)
        rows = self.selected.base_gemm_rows
        runner = ResidentRunner(
            providers={rows: self.provider},
            top_k=self.case.top_k,
            routed_scaling_factor=self.case.routed_scaling_factor,
            activation=ActivationFn.SILU,
        )
        runner.validate_plan(self.selected.plan, base_gemm_rows=rows)
        runner.plans = {
            self.phase: SelectedPlan(
                name=self.selected.name,
                base_gemm_rows=rows,
                plan=self.selected.plan,
                tiles=TileTable([(self.case.tokens, config)]),
            )
        }
        return lambda: runner.run(self.dispatch, self.batch).hidden_states

    def check(self, fn):
        import torch

        def check_output(actual):
            actual = actual.float()
            if not torch.isfinite(actual).all():
                raise ValueError("nonfinite runner output")
            relative_l2 = (
                actual - self.reference
            ).norm() / self.reference.norm().clamp_min(1e-12)
            tolerance = 0.06 if self.case.quant == "fp8" else 0.02
            if relative_l2.item() > tolerance:
                raise ValueError(
                    f"FP32 oracle relative L2 {relative_l2.item()} exceeds {tolerance}"
                )
            torch.testing.assert_close(actual, self.reference, atol=0.018, rtol=0.06)

        check_output(fn())
        if self.case.mode == "graph":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = fn()
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check_output(output)
