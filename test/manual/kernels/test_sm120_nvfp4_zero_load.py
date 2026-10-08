"""Manual real-weight SM120 NVFP4 zero-load regression and dispatch suite.

SGLANG_TEST_NVFP4_MOE_CAPTURE points at layer-1/layer-39 directories with
weights.safetensors and original sample-*.safetensors. No test dataset is
embedded. SGLANG_TEST_FLASHINFER_AUTOTUNE points at the frozen original
FlashInfer cache directory. Exact FC2 operands cover zeros, signed zeros, dense values,
alternating 32-value blocks, and the final 128-value tail.
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
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from safetensors.torch import load_file
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.gemm import sm120_nvfp4_moe_mtp_mma as op
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMoeQuantInfo,
    _run_flashinfer_cutlass,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
    and os.environ.get("SGLANG_TEST_NVFP4_MOE_CAPTURE")
    and os.environ.get("SGLANG_TEST_FLASHINFER_AUTOTUNE"),
    "requires SM120 and real NVFP4 MoE capture fixtures",
)
class TestNvfp4ZeroLoad(CustomTestCase):
    def test_runner_and_fc2_live_inputs(self):
        torch.set_num_threads(4)
        torch.manual_seed(318)
        root = Path(os.environ["SGLANG_TEST_NVFP4_MOE_CAPTURE"])
        cache = Path(os.environ["SGLANG_TEST_FLASHINFER_AUTOTUNE"])
        for path in sorted(cache.rglob("*.json")):
            self.assertTrue(AutoTuner.get().load_configs(str(path)))
        fc2 = {
            flag: load_jit(
                f"test_nvfp4_zero_fc2_{flag}",
                cuda_files=[str(Path(__file__).with_name("nvfp4_zero_load_fc2.cuh"))],
                cuda_wrappers=[
                    ("run", f"sglang::TestNvfp4ZeroFc2<{str(flag).lower()}>::run")
                ],
                extra_cuda_cflags=["-O3"],
            )
            for flag in [False, True]
        }
        full_cases = fc2_cases = 0
        for layer in [1, 39]:
            w = load_file(root / f"layer-{layer}/weights.safetensors", device="cuda")
            qs = [w[f"quant_scale_{i}"] for i in range(6)]
            samples = [
                load_file(p, device="cuda")
                for p in sorted((root / f"layer-{layer}").glob("sample-*.safetensors"))
            ]
            quant = FlashInferCutlassMoeQuantInfo(
                "fp4", w["w13"], w["w2"], quant_scales=qs, output_dtype=torch.bfloat16
            )
            config = MoeRunnerConfig(activation="relu2", is_gated=False)
            for M in [1, 2, 3, 4]:
                x = samples[0]["input"][:M].clone()
                ids = samples[0]["topk_ids"][:M].int().clone()
                tw = samples[0]["topk_weights"][:M].float().clone()
                output = {flag: torch.empty_like(x) for flag in [False, True]}
                dispatch = SimpleNamespace(
                    hidden_states=x,
                    hidden_states_scale=None,
                    topk_output=SimpleNamespace(topk_ids=ids, topk_weights=tw),
                )

                def run(
                    flag, dispatch=dispatch, quant=quant, config=config, output=output
                ):
                    with (
                        envs.SGLANG_SM120_NVFP4_ZERO_LOAD.override(flag),
                        patch.object(op, "_module", wraps=op._module) as observed,
                    ):
                        _run_flashinfer_cutlass(
                            dispatch_output=dispatch,
                            quant_info=quant,
                            runner_config=config,
                            output=output[flag],
                        )
                        observed.assert_called_once_with(flag)

                run(False)
                run(True)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run(True)
                for case in range(16):
                    if case < 3:
                        x.copy_(samples[case]["input"][:M])
                        ids.copy_(samples[case]["topk_ids"][:M])
                        tw.copy_(samples[case]["topk_weights"][:M])
                    else:
                        x.copy_(
                            torch.randn_like(x) * [0.0, 0.01, 0.1, 1.0, 3.0][case % 5]
                        )
                        ids.copy_(
                            torch.rand(M, 512, device="cuda")
                            .topk(22, dim=1)
                            .indices.int()
                        )
                        tw.copy_(torch.rand_like(tw))
                        tw.div_(tw.sum(1, keepdim=True))
                        if case == 15:
                            ids.copy_(
                                torch.arange(490, 512, device="cuda").expand_as(ids)
                            )
                    run(False)
                    run(True)
                    self.assertTrue(
                        torch.equal(
                            output[False].view(torch.int16),
                            output[True].view(torch.int16),
                        )
                    )
                    g.replay()
                    self.assertTrue(
                        torch.equal(
                            output[False].view(torch.int16),
                            output[True].view(torch.int16),
                        )
                    )
                    if case < 3 and M == 4:
                        # Fused finalize uses BF16 atomic additions: a saved full
                        # output is not a bitwise oracle, even for repeated stock
                        # calls. Compare every original FlashInfer expert term,
                        # then apply the established native slot-order FP32 sum.
                        saved = samples[case]["expected_output"][:M]
                        print(
                            "saved atomic-output unequal values",
                            layer,
                            case,
                            int((output[True] != saved).sum()),
                            flush=True,
                        )
                        original_tw = tw.clone()
                        term = torch.empty_like(x)
                        ordered = torch.zeros_like(x, dtype=torch.float32)
                        for slot in range(22):
                            tw.zero_()
                            tw[:, slot].copy_(original_tw[:, slot])
                            cutlass_fused_moe(
                                x,
                                ids,
                                tw,
                                w["w13"].view(torch.long),
                                w["w2"].view(torch.long),
                                torch.bfloat16,
                                [
                                    qs[0],
                                    qs[1].view(torch.int32),
                                    qs[2],
                                    qs[3],
                                    qs[4].view(torch.int32),
                                    qs[5],
                                ],
                                output=term,
                                activation_type=ActivationType.Relu2,
                                tune_max_num_tokens=4,
                                use_fused_finalize=True,
                            )
                            run(True)
                            print(
                                "term comparison",
                                layer,
                                case,
                                slot,
                                int((output[True] != term).sum()),
                                float(
                                    (output[True].float() - term.float()).abs().max()
                                ),
                                flush=True,
                            )
                            self.assertTrue(torch.equal(output[True], term))
                            ordered.add_(term.float())
                        tw.copy_(original_tw)
                        run(True)
                        print(
                            "ordered comparison",
                            layer,
                            case,
                            int((output[True] != ordered.bfloat16()).sum()),
                            int(
                                (
                                    output[True].view(torch.int16)
                                    != ordered.bfloat16().view(torch.int16)
                                ).sum()
                            ),
                            float((output[True].float() - ordered).abs().max()),
                            flush=True,
                        )
                        self.assertTrue(
                            torch.equal(
                                output[True].view(torch.int16),
                                ordered.bfloat16().view(torch.int16),
                            )
                        )
                    full_cases += 1
                # Native fast paths must continue to yield to the original tuner.
                with (
                    patch(
                        "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._flashinfer_autotune_active",
                        return_value=True,
                    ),
                    patch.object(
                        op,
                        "_module",
                        side_effect=AssertionError("native path during autotune"),
                    ),
                ):
                    _run_flashinfer_cutlass(
                        dispatch_output=dispatch,
                        quant_info=quant,
                        runner_config=config,
                        output=output[True],
                    )

                z = torch.empty(M * 22, 2688 // 8, dtype=torch.int32, device="cuda")
                zs = torch.full(
                    (M * 22, 2688 // 16), 56, dtype=torch.uint8, device="cuda"
                )
                partial = {
                    flag: torch.empty(M * 22, 1024, dtype=torch.bfloat16, device="cuda")
                    for flag in [False, True]
                }

                def leaf(flag, z=z, zs=zs, w=w, qs=qs, ids=ids, tw=tw, partial=partial):
                    fc2[flag].run(z, zs, w["w2"], qs[4], qs[5], ids, tw, partial[flag])

                z.zero_()
                leaf(False)
                leaf(True)
                h = torch.cuda.CUDAGraph()
                with torch.cuda.graph(h):
                    leaf(True)
                for case in range(8):
                    zs.fill_(56)
                    if case in [0, 1]:
                        z.fill_(
                            0 if case == 0 else -2004318072
                        )  # 0x88888888 signed FP4 zeros.
                    else:
                        z.random_(-2147483648, 2147483647)
                        if case == 3:
                            z.view(-1, 4)[::2].zero_()
                        elif case == 4:
                            z[
                                :, :-16
                            ].zero_()  # only the final 128 K values remain nonzero.
                        elif case == 5:
                            z[:, -16:].zero_()
                        elif case == 6:
                            zs.zero_()
                        elif case == 7:
                            zs[:, ::2].zero_()
                    leaf(False)
                    leaf(True)
                    self.assertTrue(
                        torch.equal(
                            partial[False].view(torch.int16),
                            partial[True].view(torch.int16),
                        )
                    )
                    h.replay()
                    self.assertTrue(
                        torch.equal(
                            partial[False].view(torch.int16),
                            partial[True].view(torch.int16),
                        )
                    )
                    fc2_cases += 1
                print("passed layer/M", layer, M, flush=True)
                del g, h, output, partial, run, leaf
            del w, qs, samples, quant
            gc.collect()
            torch.cuda.empty_cache()
        self.assertEqual(full_cases, 128)
        self.assertEqual(fc2_cases, 64)
        print(
            "passed full runner cases",
            full_cases,
            "FC2 operand cases",
            fc2_cases,
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
