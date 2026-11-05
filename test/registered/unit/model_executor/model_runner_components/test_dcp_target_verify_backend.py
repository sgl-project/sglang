"""Reject unsupported DCP target verify on the backend selected by speculative mode."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.model_overrides import kimi_k3  # noqa: E402
from sglang.srt.layers.attention.cutedsl_mla_backend import (  # noqa: E402
    CuteDslMLABackend,
    CuteDslMLAPrefillBackend,
)
from sglang.srt.layers.attention.hybrid_attn_backend import (  # noqa: E402
    HybridAttnBackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (  # noqa: E402
    TRTLLMMLABackend,
)
from sglang.srt.model_executor.model_runner_components import (  # noqa: E402
    attention_backend_setup as setup,
)
from sglang.srt.speculative import ragged_verify  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _hybrid(prefill_cls, decode_cls, mode):
    """The pair as the runner composes it; the children are never run."""
    hybrid = object.__new__(HybridAttnBackend)
    hybrid.prefill_backend = object.__new__(prefill_cls)
    hybrid.decode_backend = object.__new__(decode_cls)
    hybrid.spec_attn_is_decode = mode == "decode"
    hybrid.spec_attn_is_prefill = mode == "prefill"
    return hybrid


def _require(hybrid, *, mode, algorithm="DFLASH", dcp=True, unified=True):
    runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.from_string(algorithm),
    )
    with (
        patch.object(
            setup,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=dcp, dcp_size=2 if dcp else 1),
        ),
        patch.object(
            setup,
            "get_memory",
            return_value=SimpleNamespace(enable_unified_memory=unified),
        ),
        patch.object(
            setup,
            "get_spec",
            return_value=SimpleNamespace(speculative_attention_mode=mode),
        ),
    ):
        setup._require_dcp_target_verify(runner, hybrid.target_verify_backend())


def _kimi_k3_mode(algorithm: str) -> str:
    """speculative_attention_mode after Kimi-K3's DCP override."""
    cfg = SimpleNamespace(
        dcp_size=2,
        enable_symm_mem=False,
        speculative_algorithm=algorithm,
        dcp_replicate_q_proj=False,
        dcp_comm_backend="nccl",
    )
    with (
        patch.object(kimi_k3, "resolving_view", return_value=cfg),
        patch.object(
            kimi_k3, "attention_backends_of", return_value=("trtllm_mla", "cutedsl_mla")
        ),
        patch.object(kimi_k3, "_require_kimi_k3_cutedsl_dcp_support", lambda: None),
        patch.object(
            ragged_verify,
            "read_ragged_verify_mode",
            return_value=ragged_verify.RaggedVerifyMode.STATIC,
        ),
    ):
        overrides = kimi_k3._kimi_k3_overrides(object(), object())
    return overrides.get("speculative_attention_mode", "prefill")


class TestDcpTargetVerifyBackend(CustomTestCase):
    def test_bf16_prefill_mode_verifies_on_trtllm_mla_and_is_refused(self):
        """DFLASH verify on the TRTLLM MLA prefill backend must fail at setup,
        before its unsupported DCP kernel is captured or executed."""
        hybrid = _hybrid(TRTLLMMLABackend, CuteDslMLABackend, "prefill")
        with self.assertRaisesRegex(ValueError, "TRTLLMMLABackend"):
            _require(hybrid, mode="prefill")

    def test_fp8_prefill_mode_is_refused_too(self):
        """With FP8 KV the prefill side is cutedsl's prefill backend, which
        verifies with trtllm_mla's kernel and is refused the same way."""
        hybrid = _hybrid(CuteDslMLAPrefillBackend, CuteDslMLABackend, "prefill")
        with self.assertRaisesRegex(ValueError, "CuteDslMLAPrefillBackend"):
            _require(hybrid, mode="prefill")

    def test_decode_mode_verifies_on_cutedsl_mla(self):
        for prefill_cls in (TRTLLMMLABackend, CuteDslMLAPrefillBackend):
            with self.subTest(prefill=prefill_cls.__name__):
                _require(
                    _hybrid(prefill_cls, CuteDslMLABackend, "decode"), mode="decode"
                )

    def test_kimi_k3_dspark_override_verifies_on_the_decode_backend(self):
        self.assertEqual(_kimi_k3_mode("DSPARK"), "decode")
        _require(
            _hybrid(TRTLLMMLABackend, CuteDslMLABackend, _kimi_k3_mode("DSPARK")),
            mode="decode",
            algorithm="DSPARK",
        )

    def test_kimi_k3_dflash_keeps_prefill_mode_and_is_refused(self):
        mode = _kimi_k3_mode("DFLASH")
        self.assertEqual(mode, "prefill")
        with self.assertRaisesRegex(ValueError, "TRTLLMMLABackend"):
            _require(_hybrid(TRTLLMMLABackend, CuteDslMLABackend, mode), mode=mode)

    def test_only_unified_dcp_speculative_runs_are_checked(self):
        hybrid = _hybrid(TRTLLMMLABackend, CuteDslMLABackend, "prefill")
        _require(hybrid, mode="prefill", dcp=False)
        _require(hybrid, mode="prefill", unified=False)
        _require(hybrid, mode="prefill", algorithm="NONE")


if __name__ == "__main__":
    unittest.main()
