"""Manual real-weight regression for serial NVFP4 layout changes.

SGLANG_TEST_NVFP4_MOE_CAPTURE supplies layer-1/layer-39 original weight
and sample safetensors. All original weight bits and expert routes are kept.
"""

import gc
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.quantization  # noqa: F401
import torch
from flashinfer.autotuner import AutoTuner
from safetensors.torch import load_file
from sglang.kernels.ops.gemm import sm120_nvfp4_moe_mtp_mma as op
from sglang.kernels.ops.gemm.sm120_nvfp4_layout import Nvfp4LayoutCycle
from sglang.kernels.ops.gemm.sm120_nvfp4_moe_mtp_mma import nvfp4_moe_mtp_mma
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMoeQuantInfo,
    _run_flashinfer_cutlass,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components.nvfp4_layout import (
    prepare_nvfp4_layout,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
    and os.environ.get("SGLANG_TEST_NVFP4_MOE_CAPTURE"),
    "requires SM120 and original NVFP4 MoE fixtures",
)
class TestNvfp4LayoutCycle(CustomTestCase):
    def test_original_and_pair_graphs_across_layout_cycles(self):
        torch.set_num_threads(2)
        torch.manual_seed(482)
        weights = []
        samples = []
        for layer in [1, 39]:
            d = Path(os.environ["SGLANG_TEST_NVFP4_MOE_CAPTURE"]) / f"layer-{layer}"
            weights.append(load_file(d / "weights.safetensors", device="cuda"))
            samples.append(
                [
                    load_file(p, device="cuda")
                    for p in sorted(d.glob("sample-*.safetensors"))
                ]
            )
        raw = [w["w2"].view(torch.uint8).reshape(512, 1024, 1344) for w in weights]
        gold = [x.clone() for x in raw]
        manager = Nvfp4LayoutCycle(raw)
        self.assertIsNone(manager.scratch)
        manager.prepare_workspace()
        initial_ptrs = [t.data_ptr() for t in (*manager.original, *manager.pair)]
        assert manager.scratch.numel() == 704643072
        records = []
        for M in [1, 2, 3, 4]:
            inputs = []
            common = []
            out = {}
            graphs = {}
            for i, w in enumerate(weights):
                x = samples[i][0]["input"][:M].clone()
                ids = samples[i][0]["topk_ids"][:M].int().clone()
                tw = samples[i][0]["topk_weights"][:M].float().clone()
                inputs.append((x, ids, tw))
                qs = [w[f"quant_scale_{j}"] for j in range(6)]
                common.append(
                    [
                        x,
                        w["w13"],
                        qs[1],
                        None,
                        qs[4],
                        qs[2],
                        qs[5],
                        qs[0],
                        qs[3],
                        ids,
                        tw,
                    ]
                )
                for label in ["reference", "raw", "pair"]:
                    out[i, label] = torch.empty_like(x)

            def run(i, label, common=common, out=out):
                args = common[i].copy()
                args[3] = (
                    gold[i]
                    if label == "reference"
                    else manager.pair[i]
                    if label == "pair"
                    else manager.original[i]
                )
                nvfp4_moe_mtp_mma(*args, out=out[i, label], pair_layout=label == "pair")

            for pair in [False, True]:
                manager.ensure(pair)
                label = "pair" if pair else "raw"
                for i in range(2):
                    run(i, label)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        run(i, label)
                    graphs[i, label] = g
            for case in range(12):
                for i, (x, ids, tw) in enumerate(inputs):
                    if case < 3:
                        x.copy_(samples[i][case]["input"][:M])
                        ids.copy_(samples[i][case]["topk_ids"][:M])
                        tw.copy_(samples[i][case]["topk_weights"][:M])
                    else:
                        x.copy_(
                            torch.randn_like(x) * [0.0, 0.03, 0.3, 1.0, 3.0][case % 5]
                        )
                        ids.copy_(
                            torch.rand(M, 512, device="cuda")
                            .topk(22, dim=1)
                            .indices.int()
                        )
                        tw.copy_(torch.rand_like(tw))
                        tw.div_(tw.sum(1, keepdim=True))
                    run(i, "reference")
                for pair in [False, True, True, False]:
                    previous = manager.switches
                    before = manager.is_pair
                    manager.ensure(pair)
                    assert manager.switches == previous + int(pair != before)
                    label = "pair" if pair else "raw"
                    for i in range(2):
                        run(i, label)
                        torch.testing.assert_close(
                            out[i, label].view(torch.int16),
                            out[i, "reference"].view(torch.int16),
                            rtol=0,
                            atol=0,
                        )
                        graphs[i, label].replay()
                        torch.testing.assert_close(
                            out[i, label].view(torch.int16),
                            out[i, "reference"].view(torch.int16),
                            rtol=0,
                            atol=0,
                        )
                assert initial_ptrs == [
                    t.data_ptr() for t in (*manager.original, *manager.pair)
                ]
                records.append(
                    {"M": M, "case": case, "two_layers_eager_live_bitwise": True}
                )
            manager.ensure(False)
            for a, b in zip(manager.original, gold):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            print("passed M", M, "switches", manager.switches, flush=True)
            del graphs, inputs, common, out
            gc.collect()
            torch.cuda.empty_cache()

    def test_forward_modes_dispatch_and_autotune_restore(self):
        root = Path(os.environ["SGLANG_TEST_NVFP4_MOE_CAPTURE"]) / "layer-39"
        weights = load_file(root / "weights.safetensors", device="cuda")
        sample = load_file(root / "sample-0.safetensors", device="cuda")
        original = weights["w2"].view(torch.uint8).reshape(512, 1024, 1344)
        expected_bytes = original.clone()
        cycle = Nvfp4LayoutCycle([original])
        model = SimpleNamespace(_sglang_nvfp4_layout_cycle=cycle)
        qs = [weights[f"quant_scale_{i}"] for i in range(6)]
        config = MoeRunnerConfig(activation="relu2", is_gated=False)
        quant = FlashInferCutlassMoeQuantInfo(
            "fp4",
            weights["w13"],
            original,
            quant_scales=qs,
            output_dtype=torch.bfloat16,
        )
        for m in [1, 2, 3, 4]:
            x = sample["input"][:m].clone()
            dispatch = SimpleNamespace(
                hidden_states=x,
                hidden_states_scale=None,
                topk_output=SimpleNamespace(
                    topk_ids=sample["topk_ids"][:m].int().contiguous(),
                    topk_weights=sample["topk_weights"][:m].float().contiguous(),
                ),
            )
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                input_ids=torch.zeros(m, dtype=torch.int64, device="cuda"),
            )
            prepare_nvfp4_layout(model, batch)
            quant.pair_w2_weight = None
            expected = _run_flashinfer_cutlass(
                dispatch_output=dispatch, quant_info=quant, runner_config=config
            ).clone()
            for mode in [ForwardMode.TARGET_VERIFY, ForwardMode.DECODE]:
                batch.forward_mode = mode
                prepare_nvfp4_layout(model, batch)
                self.assertTrue(cycle.is_pair)
                quant.pair_w2_weight = cycle.pair[0]
                with patch.object(op, "_pair_module", wraps=op._pair_module) as calls:
                    actual = _run_flashinfer_cutlass(
                        dispatch_output=dispatch, quant_info=quant, runner_config=config
                    )
                    calls.assert_called_once_with()
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
            # Restore the original layout before the original autotuning forward.
            with patch.object(AutoTuner.get(), "is_tuning_mode", True):
                prepare_nvfp4_layout(model, batch)
                self.assertFalse(cycle.is_pair)
                self.assertTrue(torch.equal(original, expected_bytes))
                with self.assertRaisesRegex(RuntimeError, "unsupported MoE path"):
                    _run_flashinfer_cutlass(
                        dispatch_output=dispatch, quant_info=quant, runner_config=config
                    )
            quant.pair_w2_weight = None
        with envs.SGLANG_SM120_NVFP4_PHASE_LAYOUT.override(True):
            self.assertIn(
                "SGLANG_SM120_NVFP4_PHASE_LAYOUT",
                _unsupported_derived_weight_cache_error(),
            )


if __name__ == "__main__":
    unittest.main()
