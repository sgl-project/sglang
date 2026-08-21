"""Numerical tests for the varlen absorbed-MLA extend path.

Under a captured tc_piecewise prefill graph, trtllm_mla runs absorbed MLA over a
ragged q against a freshly built paged block table. These cases check the output
still matches the reference across prefix lengths, page boundaries, ragged batches
and shuffled pages.

Backends without a varlen kernel are covered by test_mla_varlen_absorbed_gate.py:
the kit builds MLA shapes (576, 512), which they reject, so every case here would
skip.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    varlen_absorbed_mla_supported,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    run_mla_attention_case,
)


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    major, minor = torch.cuda.get_device_capability()
    # Must match varlen_absorbed_mla_supported(): flashinfer resolves
    # backend="auto" to XQA when the major is not 10, and XQA rejects
    # cum_seq_lens_q. Anywhere else these cases would silently exercise the
    # FlashInfer fallback and pass without touching the code under test.
    if major != 10:
        return False, f"varlen absorbed MLA needs SM 10.x, got SM {major}.{minor}"
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()

MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
    # trtllm-gen returns bf16 under an fp8 KV cache; the kit's fp16 default
    # mismatches at the w_vc bmm.
    dtype=torch.bfloat16,
)

from sglang.test.ci.ci_register import register_cuda_ci

# 4-gpu-b200 is SM 10.0, the only per-commit runner where _supported() is true;
# 1-gpu-large (H100, SM 9.0) only exercises the skip path. Mirrors the
# registration of test_trtllm_mla.py / test_tokenspeed_mla.py.
register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


def _cases(backend: str) -> tuple:
    return (
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_zero_prefix_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_below_page_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(63,),
        ),
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_with_prefix_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(4,),
        ),
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_cross_page_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(60,),
            extend_lens=(8,),
        ),
        # Unequal lengths are what varlen exists for: a wrong cum_seq_lens_q or
        # block table only misbehaves here.
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_ragged_batch_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 64, 30),
            extend_lens=(64, 8, 33),
        ),
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_ragged_batch_32",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(0, 32, 17),
            extend_lens=(32, 5, 19),
        ),
        # Batch multiplicity seen in production: instrumenting a Kimi GSM8K run
        # showed this branch serving bs=1 30256x, bs=2 4392x, bs=3 488x and
        # bs=4 244x, while the prefill graph had only ever captured bs=1. The
        # cases above stop at bs=3, so bs=2 and bs=4 were never exercised.
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_ragged_batch2_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 96),
            extend_lens=(64, 7),
        ),
        # A single-token extend next to a long one is the widest spread
        # cum_seq_lens_q has to describe, and the shape a decode-tail request
        # takes when it lands in the same prefill batch.
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_ragged_batch4_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 64, 30, 128),
            extend_lens=(64, 8, 33, 1),
        ),
        MLAAttentionCase(
            name=f"pcg_extend_{backend}_ragged_batch4_32",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(0, 32, 17, 96),
            extend_lens=(32, 5, 19, 1),
        ),
    )


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTRTLLMMLAPiecewiseExtend(CustomTestCase):
    CASES = _cases("trtllm_mla")

    def test_piecewise_extend_matches_reference(self):
        for case in self.CASES:
            with self.subTest(case=case.name):
                fixture = run_mla_attention_case(
                    self,
                    case,
                    piecewise=True,
                    fp8_kv_cache=True,
                    atol=2e-1,
                    rtol=2e-1,
                    **MLA_SHAPE_KWARGS,
                )
                # block_kv_indices is populated only when the varlen absorbed path
                # is selected. Without this, any future gate that quietly reroutes
                # to the FlashInfer fallback would still make these cases pass.
                self.assertIsNotNone(
                    fixture.backend.forward_prefill_metadata.block_kv_indices,
                    "varlen absorbed MLA did not run; this case validated the "
                    "FlashInfer fallback instead",
                )

    def test_optout_backend_keeps_the_paged_fallback(self):
        # Negative half of the routing gate: the cases above prove the varlen
        # path is taken when the capability is declared, this proves the
        # FlashInfer fallback is kept when it is not. That is the contract
        # cutedsl_mla and tokenspeed_mla rely on.
        case = self.CASES[0]
        with patch.object(TRTLLMMLABackend, "supports_varlen_absorbed_mla", False):
            fixture = run_mla_attention_case(
                self,
                case,
                piecewise=True,
                fp8_kv_cache=True,
                atol=2e-1,
                rtol=2e-1,
                **MLA_SHAPE_KWARGS,
            )
        self.assertIsNone(
            fixture.backend.forward_prefill_metadata.block_kv_indices,
            "an opted-out backend still took the varlen absorbed path",
        )

    def test_unsupported_env_keeps_the_paged_fallback(self):
        # varlen_absorbed_mla_supported() is what rules out FP4 KV and non-SM10
        # devices, and ServerArgs reads the same helper to decide whether the
        # prefill graph may be tc_piecewise at all. This pins the runtime half of
        # that wiring: helper says no -> __init__ records it -> forward_extend
        # keeps the paged fallback. The fixture cannot build an FP4 KV pool, so
        # the dtype half is pinned in test_mla_varlen_absorbed_gate.py instead.
        case = self.CASES[0]
        with patch(
            "sglang.srt.layers.attention.trtllm_mla_backend."
            "varlen_absorbed_mla_supported",
            return_value=False,
        ):
            fixture = run_mla_attention_case(
                self,
                case,
                piecewise=True,
                fp8_kv_cache=True,
                atol=2e-1,
                rtol=2e-1,
                **MLA_SHAPE_KWARGS,
            )
        self.assertIsNone(
            fixture.backend.forward_prefill_metadata.block_kv_indices,
            "the varlen absorbed path ran despite an unsupported environment",
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestVarlenAbsorbedArchGate(CustomTestCase):
    """The arch gate must be asserted, not merely skipped over.

    On SM != 10.x flashinfer resolves backend="auto" to XQA, which rejects
    cum_seq_lens_q. The class above skips there, so without this test the
    non-SM10 half of the gate would have no coverage at all -- and that gate
    exists because trusting the ``backend == "trtllm-gen"`` string was wrong.
    1-gpu-large (H100, SM 9.0) is a per-commit runner, so this runs every PR.
    """

    def test_arch_gate_matches_flashinfer_resolution(self):
        major, minor = torch.cuda.get_device_capability()
        expected = major == 10
        # Call the predicate rather than instantiating a backend (construction
        # needs a full ModelRunner). fp8 KV is the shipped configuration, so this
        # isolates the arch half of the gate from the FP4-KV half.
        self.assertEqual(
            varlen_absorbed_mla_supported(torch.float8_e4m3fn),
            expected,
            f"the arch gate disagrees with SM {major}.{minor}",
        )
        if expected:
            self.assertTrue(
                _SUPPORTED,
                f"SM {major}.{minor} is SM 10.x, so the numerical cases above "
                "must not be skipped",
            )
        else:
            self.assertFalse(
                _SUPPORTED,
                f"SM {major}.{minor} must take the FlashInfer fallback, "
                "not varlen absorbed MLA",
            )


if __name__ == "__main__":
    unittest.main()
