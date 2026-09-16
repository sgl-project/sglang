"""Shared numerical cases for varlen absorbed-MLA extend."""

from __future__ import annotations

from unittest.mock import patch

import torch

from sglang.srt.environ import envs
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
    # FlashInfer routes non-SM10 auto calls to XQA, which rejects variable Q.
    if major != 10:
        return False, f"varlen absorbed MLA needs SM 10.x, got SM {major}.{minor}"
    return True, ""


MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
    # trtllm-gen returns bf16 for FP8 KV.
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
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_ragged_batch2_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 96),
            extend_lens=(64, 7),
        ),
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
        # Covers FlashInfer's TRTLLM-GEN head-count gap.
        MLAAttentionCase(
            name=f"{prefix}_extend_{backend}_cute_fallback_heads96_64",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=96,
            page_size=64,
            prefix_lens=(0, 64, 30),
            extend_lens=(64, 8, 1),
        ),
    )


class VarlenAbsorbedExtendMixin:
    """Assertions shared by piecewise and breakable capture tests."""

    CASES: tuple = ()
    MODE_KWARGS: dict = {}
    MODE_NAME: str = ""

    def _assert_wide_auto_fallback(self, run_decode):
        import flashinfer

        varlen_calls = [
            call
            for call in run_decode.call_args_list
            if call.kwargs.get("cum_seq_lens_q") is not None
        ]
        self.assertGreater(len(varlen_calls), 0)
        forced_trt_kwargs = dict(varlen_calls[-1].kwargs)
        forced_trt_kwargs["backend"] = "trtllm-gen"
        with self.assertRaises(ValueError):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(**forced_trt_kwargs)

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
                self.assertIsNotNone(
                    fixture.backend.forward_prefill_metadata.block_kv_indices,
                    f"varlen absorbed MLA did not run under {self.MODE_NAME} "
                    "capture; this case validated the FlashInfer fallback instead",
                )

    def test_non_owner_keeps_the_paged_fallback(self):
        case = self.CASES[0]
        with patch.object(TRTLLMMLABackend, "owns_varlen_absorbed_extend", False):
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
        import flashinfer

        case = self.CASES[0]
        decode = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
        with patch.object(
            flashinfer.decode,
            "trtllm_batch_decode_with_kv_cache_mla",
            wraps=decode,
        ) as run_decode:
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
        varlen_calls = [
            call
            for call in run_decode.call_args_list
            if call.kwargs.get("cum_seq_lens_q") is not None
        ]
        self.assertGreater(len(varlen_calls), 0)
        self.assertTrue(
            all(
                "multi_ctas_kv_counter_buffer" not in call.kwargs
                for call in varlen_calls
            ),
            "SGLang passed its dense-decode counter to a variable-Q call",
        )

    def test_cute_fallback_survives_real_capture_replay(self):
        import flashinfer

        case = self.CASES[-1]
        self.assertEqual(case.num_heads, 96)
        decode = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
        with patch.object(
            flashinfer.decode,
            "trtllm_batch_decode_with_kv_cache_mla",
            wraps=decode,
        ) as run_decode:
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
            f"wide-head MLA did not run the varlen absorbed path under real "
            f"{self.MODE_NAME} capture/replay",
        )
        self._assert_wide_auto_fallback(run_decode)

    def test_cute_fallback_supports_bf16_kv(self):
        import flashinfer

        case = self.CASES[-1]
        decode = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
        with patch.object(
            flashinfer.decode,
            "trtllm_batch_decode_with_kv_cache_mla",
            wraps=decode,
        ) as run_decode:
            fixture = run_mla_attention_case(
                self,
                case,
                fp8_kv_cache=False,
                atol=2e-1,
                rtol=2e-1,
                **self.MODE_KWARGS,
                **MLA_SHAPE_KWARGS,
            )
        self.assertIsNotNone(fixture.backend.forward_prefill_metadata.block_kv_indices)
        self._assert_wide_auto_fallback(run_decode)

    def test_skip_softmax_uses_paged_fallback(self):
        case = self.CASES[0]
        with envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.override(1.0):
            fixture = run_mla_attention_case(
                self,
                case,
                fp8_kv_cache=True,
                atol=2e-1,
                rtol=2e-1,
                **self.MODE_KWARGS,
                **MLA_SHAPE_KWARGS,
            )
        self.assertIsNone(fixture.backend.forward_prefill_metadata.block_kv_indices)
        self.assertTrue(
            fixture.backend.forward_prefill_metadata.fallback_to_flashinfer_impl
        )

    def test_shared_workspace_survives_backend_switch(self):
        low_before = run_mla_attention_case(
            self,
            self.CASES[0],
            fp8_kv_cache=True,
            atol=2e-1,
            rtol=2e-1,
            **self.MODE_KWARGS,
            **MLA_SHAPE_KWARGS,
        )
        wide = run_mla_attention_case(
            self,
            self.CASES[-1],
            fp8_kv_cache=True,
            atol=2e-1,
            rtol=2e-1,
            **self.MODE_KWARGS,
            **MLA_SHAPE_KWARGS,
        )
        cute = run_mla_attention_case(
            self,
            MLAAttentionCase(
                name=f"{self.MODE_NAME}_dense_cutedsl_workspace_writer",
                backend="cutedsl_mla",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                page_size=64,
                prefix_lens=(63,),
            ),
            fp8_kv_cache=False,
            atol=2e-1,
            rtol=2e-1,
            **self.MODE_KWARGS,
            **MLA_SHAPE_KWARGS,
        )
        low_after = run_mla_attention_case(
            self,
            self.CASES[0],
            fp8_kv_cache=True,
            atol=2e-1,
            rtol=2e-1,
            **self.MODE_KWARGS,
            **MLA_SHAPE_KWARGS,
        )

        shared = low_before.backend._varlen_absorbed_workspace_buffer
        self.assertTrue(
            all(
                fixture.backend.backend == "trtllm-gen"
                for fixture in (low_before, wide, low_after)
            )
        )
        self.assertIs(wide.backend._varlen_absorbed_workspace_buffer, shared)
        self.assertIs(low_after.backend._varlen_absorbed_workspace_buffer, shared)
        self.assertEqual(cute.backend.backend, "cute-dsl")
        self.assertIsNot(cute.backend.workspace_buffer, shared)
        for fixture in (low_before, wide, low_after):
            self.assertIsNot(shared, fixture.backend.workspace_buffer)


__all__ = [
    "supported",
    "MLA_SHAPE_KWARGS",
    "cases",
    "VarlenAbsorbedExtendMixin",
]
