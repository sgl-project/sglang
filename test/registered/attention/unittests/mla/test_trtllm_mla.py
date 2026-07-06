import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.srt.utils import is_sm100_supported, is_sm120_supported
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    run_mla_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_mla_eagle_verify_case,
    run_mla_eagle_verify_cuda_graph_case,
)

# trtllm_mla goes through FlashInfer's XQA MLA path. Per PLAN.md and the
# project's is_sm120_supported helper (device_capability_majors=[12]), the
# decode path requires SM120a / SM121a (Blackwell variants), i.e. major==12.
# The backend itself has no hard gate — failure surfaces inside FlashInfer at
# kernel-dispatch time — so we mirror is_sm120_supported here.
_REQUIRED_MAJOR = 12

MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
)


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    major, minor = torch.cuda.get_device_capability()
    if major != _REQUIRED_MAJOR:
        return (
            False,
            f"trtllm_mla requires SM 12.0a / 12.1a (FlashInfer XQA MLA), "
            f"got SM {major}.{minor}",
        )
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


def _cutedsl_supported() -> tuple[bool, str]:
    # The cuteDSL MLA decode kernel (cutedsl_mla backend) runs on SM100 (B200)
    # and SM120/SM121 (Blackwell variants). The topk>1 tree-verify cascade also
    # needs the flashinfer non-causal toggle (flashinfer-ai/flashinfer #3771);
    # that dependency is probed separately by _causal_toggle_supported() so the
    # topk>1 cases SKIP (not error) on an older flashinfer, while the topk==1
    # chain case -- which does not use causal= -- still runs.
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if not (is_sm100_supported() or is_sm120_supported()):
        major, minor = torch.cuda.get_device_capability()
        return (
            False,
            f"cutedsl_mla requires SM 10.0 (B200) / 12.0a / 12.1a, "
            f"got SM {major}.{minor}",
        )
    return True, ""


_CUTEDSL_SUPPORTED, _CUTEDSL_SKIP_REASON = _cutedsl_supported()


def _causal_toggle_supported() -> bool:
    """True iff the installed flashinfer exposes the cuteDSL non-causal toggle
    (``cute_dsl_mla_decode(causal=...)``, flashinfer-ai/flashinfer #3771), which
    the topk>1 tree-verify cascade requires. Lets the topk>1 cases SKIP instead
    of erroring on an older flashinfer."""
    try:
        import inspect

        # The public flashinfer.cute_dsl.attention.cute_dsl_mla_decode is a
        # ``*args, **kwargs`` dispatcher (mla_dispatch.py), so its signature does
        # NOT list ``causal``; the toggle lives on the monolithic implementation.
        from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
            cute_dsl_mla_decode as _mono_mla_decode,
        )

        return "causal" in inspect.signature(_mono_mla_decode).parameters
    except Exception:
        return False


_CAUSAL_TOGGLE_OK = _causal_toggle_supported()


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTRTLLMMLAAttentionBackendCorrectness(CustomTestCase):
    # trtllm_mla allows page_size in {32, 64} (server_args.py:2790-2794).
    # Cover both, with extend + decode + ragged + page-boundary layouts.
    CASES = (
        # ----- page_size=64 -----
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_exact_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_below_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(63,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_above_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(65,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_prefix_exact_page_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(4,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_cross_page_boundary_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(63,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_ragged_page_boundary_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 32, 64),
            extend_lens=(63, 32, 1),
        ),
        MLAAttentionCase(
            name="mla_decode_trtllm_page_boundary_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        MLAAttentionCase(
            name="mla_decode_trtllm_bsz1_nonzero_prefix_64",
            backend="trtllm_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(31,),
        ),
        # ----- page_size=32 -----
        MLAAttentionCase(
            name="mla_extend_trtllm_zero_prefix_exact_page_32",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(0,),
            extend_lens=(32,),
        ),
        MLAAttentionCase(
            name="mla_extend_trtllm_cross_page_boundary_32",
            backend="trtllm_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_decode_trtllm_page_boundary_32",
            backend="trtllm_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=32,
            prefix_lens=(30, 31, 32),
        ),
    )

    def test_projected_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mla_attention_case(self, case, **MLA_SHAPE_KWARGS)


@unittest.skipIf(not _CUTEDSL_SUPPORTED, _CUTEDSL_SKIP_REASON)
class TestCuteDSLMLAEagleVerifyTreeCorrectness(CustomTestCase):
    """EAGLE target-verify on the cuteDSL MLA backend (cutedsl_mla).

    The topk>1 (tree) verify runs the 2-pass cascade in
    ``TRTLLMMLABackend._forward_tree_verify`` (the cuteDSL fold decode kernel
    returns LSE, which trtllm-gen does not). The topk==1 (chain) case is a
    regression-equivalence guard that exercises the unchanged single-pass path
    through the same fixture. Mirrors the Triton tree case in
    ``test_triton.py`` (parent_indices == (-1, 0, 0), draft_token_num == 3).
    """

    # The cuteDSL decode kernel needs paged KV with page_size in {32, 64}
    # (it rejects page_size == 1), and the prefix/draft dims must be the
    # production MLA shape (kv_lora_rank=512, qk_rope_head_dim=64).
    EAGLE_VERIFY_CASES = (
        (
            MLAAttentionCase(
                name="runner_eagle_verify_cutedsl_mla_chain",
                backend="cutedsl_mla",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=64,
                prefix_lens=(64, 70),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            MLAAttentionCase(
                name="runner_eagle_verify_cutedsl_mla_tree",
                backend="cutedsl_mla",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=64,
                prefix_lens=(64, 70),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            MLAAttentionCase(
                name="runner_cuda_graph_eagle_verify_cutedsl_mla_chain",
                backend="cutedsl_mla",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=64,
                prefix_lens=(64, 70),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            MLAAttentionCase(
                name="runner_cuda_graph_eagle_verify_cutedsl_mla_tree",
                backend="cutedsl_mla",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                page_size=64,
                prefix_lens=(64, 70),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
    )

    def test_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(case=case.name, topk=topk, spec_kind=spec_kind):
                if topk > 1 and not _CAUSAL_TOGGLE_OK:
                    self.skipTest(
                        "topk>1 tree-verify needs flashinfer cute_dsl_mla_decode "
                        "causal= toggle (flashinfer-ai/flashinfer #3771)"
                    )
                run_mla_eagle_verify_case(
                    self, case, topk=topk, spec_kind=spec_kind, **MLA_SHAPE_KWARGS
                )

    def test_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, topk=topk, spec_kind=spec_kind):
                if topk > 1 and not _CAUSAL_TOGGLE_OK:
                    self.skipTest(
                        "topk>1 tree-verify needs flashinfer cute_dsl_mla_decode "
                        "causal= toggle (flashinfer-ai/flashinfer #3771)"
                    )
                run_mla_eagle_verify_cuda_graph_case(
                    self, case, topk=topk, spec_kind=spec_kind, **MLA_SHAPE_KWARGS
                )


if __name__ == "__main__":
    unittest.main()
