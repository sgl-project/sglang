"""Shared numerical-test scaffolding for the varlen absorbed-MLA extend path.

test_trtllm_mla_piecewise.py and test_trtllm_mla_breakable.py (both under
test/registered/attention/unittests/mla/) exercise the same cases and
assertions under the two halves of TRTLLMMLABackend's use_varlen_absorbed
predicate (is_in_tc_piecewise_cuda_graph() vs is_in_breakable_cuda_graph()).
Only the capture-mode kwarg and the case-name prefix differ, so both files
import from here instead of maintaining two copies of the same case list and
assertions.

Lives under python/sglang/test/kits/ rather than test/registered/: files
under test/registered/ must each carry a CI registry call
(scripts/lint/check_registered_tests.py enforces this), which does not fit a
shared-helper module with no TestCase classes of its own -- the same reason
run_mla_attention_case() etc. live in mla_attention.py in this same
directory rather than in test/registered/.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    run_mla_attention_case,
    run_mla_attention_case_captured,
)


def supported() -> tuple[bool, str]:
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


MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
    # trtllm-gen returns bf16 under an fp8 KV cache; the kit's fp16 default
    # mismatches at the w_vc bmm.
    dtype=torch.bfloat16,
)


def cases(backend: str, prefix: str) -> tuple:
    return (
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_zero_prefix_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_below_page_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(63,),
        ),
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_with_prefix_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(4,),
        ),
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_cross_page_64",
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
            name=f"{prefix}_extend_{backend}_ragged_batch_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 64, 30),
            extend_lens=(64, 8, 33),
        ),
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_ragged_batch_32",
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
            name=f"{prefix}_extend_{backend}_ragged_batch2_64",
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
            name=f"{prefix}_extend_{backend}_ragged_batch4_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 64, 30, 128),
            extend_lens=(64, 8, 33, 1),
        ),
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_ragged_batch4_32",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(0, 32, 17, 96),
            extend_lens=(32, 5, 19, 1),
        ),
    )


class VarlenAbsorbedExtendMixin:
    """Shared assertions for the two capture modes. Not a TestCase subclass
    itself -- pytest's unittest plugin collects every TestCase subclass
    regardless of name, so mixing this into a concrete
    class Foo(VarlenAbsorbedExtendMixin, CustomTestCase) is what keeps this
    shared code from also running (with an empty CASES) as its own test.
    Concrete subclasses set CASES and MODE_KWARGS
    ({"piecewise": True} or {"breakable": True})."""

    CASES: tuple = ()
    MODE_KWARGS: dict = {}
    # Included in assertion messages so a failure names which capture mode
    # was under test without needing to look at the file it came from.
    MODE_NAME: str = ""

    def test_extend_matches_reference(self):
        for case in self.CASES:
            with self.subTest(case=case.name):
                fixture = run_mla_attention_case(
                    self,
                    case,
                    fp8_kv_cache=True,
                    atol=2e-1,
                    rtol=2e-1,
                    **self.MODE_KWARGS,
                    **MLA_SHAPE_KWARGS,
                )
                # block_kv_indices is populated only when the varlen absorbed path
                # is selected. Without this, any future gate that quietly reroutes
                # to the FlashInfer fallback would still make these cases pass.
                self.assertIsNotNone(
                    fixture.backend.forward_prefill_metadata.block_kv_indices,
                    f"varlen absorbed MLA did not run under {self.MODE_NAME} "
                    "capture; this case validated the FlashInfer fallback instead",
                )

    def test_optout_backend_keeps_the_paged_fallback(self):
        # Negative half of the routing gate: the cases above prove the varlen
        # path is taken when the capability is declared, this proves the
        # FlashInfer fallback is kept when it is not. That is the contract
        # cutedsl_mla and tokenspeed_mla rely on, and it must hold under both
        # capture modes.
        case = self.CASES[0]
        with patch.object(TRTLLMMLABackend, "supports_varlen_absorbed_mla", False):
            fixture = run_mla_attention_case(
                self,
                case,
                fp8_kv_cache=True,
                atol=2e-1,
                rtol=2e-1,
                **self.MODE_KWARGS,
                **MLA_SHAPE_KWARGS,
            )
        self.assertIsNone(
            fixture.backend.forward_prefill_metadata.block_kv_indices,
            f"an opted-out backend still took the varlen absorbed path under "
            f"{self.MODE_NAME} capture",
        )

    def test_unsupported_env_keeps_the_paged_fallback(self):
        # varlen_absorbed_mla_supported() is what rules out FP4 KV and non-SM10
        # devices, and ServerArgs reads the same helper to decide whether the
        # prefill graph may serve the fast path at all. This pins the runtime
        # half of that wiring under this capture mode; the dtype half is pinned
        # in test_mla_varlen_absorbed_gate.py.
        case = self.CASES[0]
        with patch(
            "sglang.srt.layers.attention.trtllm_mla_backend."
            "varlen_absorbed_mla_supported",
            return_value=False,
        ):
            fixture = run_mla_attention_case(
                self,
                case,
                fp8_kv_cache=True,
                atol=2e-1,
                rtol=2e-1,
                **self.MODE_KWARGS,
                **MLA_SHAPE_KWARGS,
            )
        self.assertIsNone(
            fixture.backend.forward_prefill_metadata.block_kv_indices,
            f"the varlen absorbed path ran under {self.MODE_NAME} capture "
            "despite an unsupported environment",
        )

    def test_extend_survives_real_capture_replay(self):
        # The cases above only set the capture-mode flag; this proves the
        # varlen absorbed MLA metadata is still valid after a real graph
        # replay, not just in a plain eager call.
        case = self.CASES[0]
        fixture = run_mla_attention_case_captured(
            self,
            case,
            fp8_kv_cache=True,
            atol=2e-1,
            rtol=2e-1,
            **self.MODE_KWARGS,
            **MLA_SHAPE_KWARGS,
        )
        self.assertIsNotNone(
            fixture.backend.forward_prefill_metadata.block_kv_indices,
            f"varlen absorbed MLA did not run under real {self.MODE_NAME} "
            "capture/replay; this case validated the FlashInfer fallback instead",
        )


__all__ = [
    "supported",
    "MLA_SHAPE_KWARGS",
    "cases",
    "VarlenAbsorbedExtendMixin",
]
