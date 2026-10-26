"""Each attention backend declares the speculative combinations it implements,
and the runner that builds it for a speculative forward refuses any other.

A combination is the forward (multi-step draft decode, draft extend, target
verify), whether the model has sliding-window layers, whether the runner
captures CUDA graphs, and whether its KV storage is translated (the unified
pool, or a draft fused into it). The backends are built through the runner's
and the draft factory's own paths; their constructors are stood in for, as the
check reads only the class and the runner.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import base_attn_backend  # noqa: E402
from sglang.srt.layers.attention import hybrid_attn_backend  # noqa: E402
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend  # noqa: E402
from sglang.srt.layers.attention.base_attn_backend import (  # noqa: E402
    SpecCombination,
    SpecMode,
    spec_combinations,
)
from sglang.srt.layers.attention.cutedsl_mla_backend import (  # noqa: E402
    CuteDslMLABackend,
)
from sglang.srt.layers.attention.flashattention_backend import (  # noqa: E402
    FlashAttentionBackend,
)
from sglang.srt.layers.attention.flashinfer_backend import (  # noqa: E402
    FlashInferAttnBackend,
)
from sglang.srt.layers.attention.flashinfer_mla_backend import (  # noqa: E402
    FlashInferMLAAttnBackend,
)
from sglang.srt.layers.attention.intel_amx_backend import (  # noqa: E402
    IntelAMXAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  # noqa: E402
from sglang.srt.layers.attention.trtllm_mha_backend import (  # noqa: E402
    TRTLLMHAAttnBackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (  # noqa: E402
    TRTLLMMLABackend,
)
from sglang.srt.model_executor.model_runner_components import (  # noqa: E402
    attention_backend_setup as setup,
)
from sglang.srt.speculative import draft_utils  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TRANSLATED = spec_combinations(*SpecMode, translated=(True,))

# What each backend implements on translated storage.
_DECLARED = {
    TritonAttnBackend: _TRANSLATED,
    FlashAttentionBackend: _TRANSLATED,
    FlashInferAttnBackend: spec_combinations(SpecMode.DRAFT_DECODE, translated=(True,))
    | spec_combinations(SpecMode.DRAFT_EXTEND, window=(False,), translated=(True,))
    | spec_combinations(
        SpecMode.DRAFT_EXTEND, window=(True,), cuda_graph=(False,), translated=(True,)
    )
    | spec_combinations(SpecMode.TARGET_VERIFY, window=(False,), translated=(True,)),
    TRTLLMHAAttnBackend: spec_combinations(
        SpecMode.TARGET_VERIFY, SpecMode.DRAFT_DECODE, translated=(True,)
    )
    | spec_combinations(SpecMode.DRAFT_EXTEND, window=(False,), translated=(True,)),
}
_MLA_VERIFY = spec_combinations(
    SpecMode.TARGET_VERIFY, window=(False,), translated=(True,)
)
# The MLA family (flashmla derives from it too): a fused draft is MHA, so only
# the target reads translated storage here.
for _cls in (FlashInferMLAAttnBackend, TRTLLMMLABackend, CuteDslMLABackend):
    _DECLARED[_cls] = _MLA_VERIFY
# Static storage only.
for _cls in (AiterAttnBackend, IntelAMXAttnBackend):
    _DECLARED[_cls] = frozenset()


def _runner(
    combination: SpecCombination,
    *,
    draft: bool = False,
    algorithm: SpeculativeAlgorithm = SpeculativeAlgorithm.EAGLE,
    **extra,
):
    return SimpleNamespace(
        sliding_window_size=128 if combination.window else None,
        kv_index_translator=SimpleNamespace(is_translating=combination.translated),
        is_draft_worker=draft,
        spec_algorithm=algorithm,
        draft_attention_backend=None,
        **extra,
    )


def _graphs(combination: SpecCombination):
    return patch.object(
        base_attn_backend,
        "get_exec",
        return_value=SimpleNamespace(
            graph=SimpleNamespace(disable_cuda_graph=not combination.cuda_graph)
        ),
    )


def _built(cls):
    """An instance of ``cls`` as its constructor would leave its class."""
    return object.__new__(cls)


def _combination(mode, *, window=True, cuda_graph=True, translated=True):
    return SpecCombination(
        mode=mode, window=window, cuda_graph=cuda_graph, translated=translated
    )


class TestDeclarations(unittest.TestCase):
    def test_each_backend_accepts_what_it_declares_and_refuses_the_rest(self):
        """Every combination on static storage, as before; on translated
        storage only what the backend declares, refused with a message that
        names the backend and the combination."""
        for cls, declared in _DECLARED.items():
            for combination in spec_combinations(*SpecMode):
                with self.subTest(backend=cls.__name__, combo=combination.describe()):
                    runner = _runner(combination)
                    with _graphs(combination):
                        if not combination.translated or combination in declared:
                            _built(cls).require_spec_mode(runner, combination.mode)
                            continue
                        with self.assertRaises(ValueError) as refused:
                            _built(cls).require_spec_mode(runner, combination.mode)
                    message = str(refused.exception)
                    self.assertIn(cls.__name__, message)
                    self.assertIn(combination.describe(), message)


class _Container:
    """A multi-step draft backend: one step backend per draft step."""

    def __init__(self, cls, steps=2):
        self.attn_backends = [_built(cls) for _ in range(steps)]


class TestDraftFactory(unittest.TestCase):
    """The draft factory checks what it builds: every step of a multi-step
    draft decode, and the draft-extend backend."""

    def _decode(self, cls, name, combination):
        runner = _runner(combination, draft=True)
        container = _Container(cls)
        factory = draft_utils.DraftBackendFactory(
            runner, topk=1, speculative_num_steps=3
        )
        with (
            _graphs(combination),
            patch.object(draft_utils, "attention_backends", return_value=(name, name)),
            patch.object(
                draft_utils.DraftBackendFactory,
                f"_create_{name}_decode_backend",
                lambda self: (name, container),
            ),
            patch.object(
                draft_utils, "_assert_draft_needs_no_conv_sidecar", lambda r: None
            ),
            patch(
                "sglang.srt.layers.attention.attention_registry."
                "attn_backend_wrapper_for_draft_decode",
                side_effect=lambda r, b: b,
            ),
        ):
            return factory.create_decode_backend()

    def _extend(self, cls, name, combination):
        runner = _runner(combination, draft=True)
        factory = draft_utils.DraftBackendFactory(
            runner, topk=1, speculative_num_steps=3
        )
        with (
            _graphs(combination),
            patch.object(draft_utils, "attention_backends", return_value=(name, name)),
            patch.object(
                draft_utils,
                "get_spec",
                return_value=SimpleNamespace(speculative_attention_mode="prefill"),
            ),
            patch.object(
                draft_utils.DraftBackendFactory,
                f"_create_{name}_prefill_backend",
                lambda self: (name, _built(cls)),
            ),
            patch(
                "sglang.srt.layers.attention.attention_registry."
                "attn_backend_wrapper_for_draft_extend",
                side_effect=lambda r, b: b,
            ),
        ):
            return factory.create_draft_extend_backend()

    def test_aiter_refuses_a_fused_window_draft(self):
        """BUG REGRESSION (from the window-only and mixed fused drafts on
        aiter). With the draft backend lists gone, aiter steps were built for a
        draft fused into the target's pool, whose window ids aiter cannot
        read. The factory now refuses it, naming the combination."""
        combination = _combination(SpecMode.DRAFT_DECODE)
        with self.assertRaisesRegex(
            ValueError, "AiterAttnBackend.*multi-step draft decode with sliding"
        ):
            self._decode(AiterAttnBackend, "aiter", combination)
        static = _combination(SpecMode.DRAFT_DECODE, translated=False)
        self.assertIsNotNone(self._decode(AiterAttnBackend, "aiter", static))

    def test_window_drafts_build_on_the_backends_that_read_their_rails(self):
        for cls, name in (
            (TritonAttnBackend, "triton"),
            (FlashInferAttnBackend, "flashinfer"),
            (TRTLLMHAAttnBackend, "trtllm_mha"),
            (FlashAttentionBackend, "fa3"),
        ):
            with self.subTest(backend=name):
                built = self._decode(cls, name, _combination(SpecMode.DRAFT_DECODE))
                self.assertEqual(len(built.attn_backends), 2)

    def test_flashinfer_window_draft_extend_builds_eager_only(self):
        """BUG REGRESSION. A graph replay of a FlashInfer draft extend plans
        its window wrapper without the window (window_left -1), so the kernel
        has no per-query window mask; the combination was declared anyway.
        The eager extend, planned with the window, builds; with CUDA graphs it
        is refused at init."""
        eager = _combination(SpecMode.DRAFT_EXTEND, cuda_graph=False)
        built = self._extend(FlashInferAttnBackend, "flashinfer", eager)
        self.assertIsInstance(built, FlashInferAttnBackend)
        graph = _combination(SpecMode.DRAFT_EXTEND)
        with self.assertRaisesRegex(
            ValueError,
            "FlashInferAttnBackend.*draft extend with sliding-window layers, "
            "with CUDA graphs",
        ):
            self._extend(FlashInferAttnBackend, "flashinfer", graph)
        no_window = _combination(SpecMode.DRAFT_EXTEND, window=False)
        self._extend(FlashInferAttnBackend, "flashinfer", no_window)

    def test_trtllm_mha_refuses_a_fused_window_draft_extend(self):
        combination = _combination(SpecMode.DRAFT_EXTEND)
        with self.assertRaisesRegex(ValueError, "TRTLLMHAAttnBackend.*draft extend"):
            self._extend(TRTLLMHAAttnBackend, "trtllm_mha", combination)
        no_window = _combination(SpecMode.DRAFT_EXTEND, window=False)
        self._extend(TRTLLMHAAttnBackend, "trtllm_mha", no_window)


class TestRunnerBackend(unittest.TestCase):
    """A speculative runner checks its own backend: target verify on the
    target, draft extend on a draft; a split pair answers with the child
    that runs the forward."""

    def _build(
        self,
        prefill_cls,
        decode_cls,
        combination,
        *,
        draft=False,
        mode="prefill",
        algorithm=SpeculativeAlgorithm.EAGLE,
    ):
        runner = _runner(
            combination,
            draft=draft,
            algorithm=algorithm,
            model_config=SimpleNamespace(context_len=2048),
            kv_cache_dtype=None,
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            init_new_workspace=None,
        )
        constructors = {
            "prefill-test": lambda r: _built(prefill_cls),
            "decode-test": lambda r: _built(decode_cls),
        }
        resolved = setup.ResolvedAttentionBackendStr(
            prefill="prefill-test",
            decode="decode-test" if decode_cls is not prefill_cls else "prefill-test",
        )
        with (
            _graphs(combination),
            patch.dict(setup.ATTENTION_BACKENDS, constructors),
            patch.object(setup, "attn_backend_wrapper", side_effect=lambda r, b: b),
            patch.object(
                setup,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
            patch.object(
                hybrid_attn_backend,
                "get_spec",
                return_value=SimpleNamespace(speculative_attention_mode=mode),
            ),
        ):
            return setup._build_resolved_backend(
                model_runner=runner, resolved=resolved, init_new_workspace=False
            )

    def test_flashinfer_refuses_a_translated_window_target_verify(self):
        combination = _combination(SpecMode.TARGET_VERIFY)
        with self.assertRaisesRegex(
            ValueError, "FlashInferAttnBackend.*target verify with sliding"
        ):
            self._build(FlashInferAttnBackend, FlashInferAttnBackend, combination)
        no_window = _combination(SpecMode.TARGET_VERIFY, window=False)
        self._build(FlashInferAttnBackend, FlashInferAttnBackend, no_window)

    def test_a_split_pair_checks_the_child_that_verifies(self):
        combination = _combination(SpecMode.TARGET_VERIFY)
        # Verify on the prefill side (triton) builds; on the decode side
        # (flashinfer) it is refused.
        self._build(TritonAttnBackend, FlashInferAttnBackend, combination)
        with self.assertRaisesRegex(ValueError, "FlashInferAttnBackend"):
            self._build(
                TritonAttnBackend, FlashInferAttnBackend, combination, mode="decode"
            )

    def test_a_draft_runner_checks_draft_extend(self):
        combination = _combination(SpecMode.DRAFT_EXTEND)
        with self.assertRaisesRegex(ValueError, "AiterAttnBackend.*draft extend"):
            self._build(AiterAttnBackend, AiterAttnBackend, combination, draft=True)
        self._build(TritonAttnBackend, TritonAttnBackend, combination, draft=True)

    def test_a_runner_without_speculation_is_not_checked(self):
        combination = _combination(SpecMode.TARGET_VERIFY)
        self._build(
            AiterAttnBackend,
            AiterAttnBackend,
            combination,
            algorithm=SpeculativeAlgorithm.NONE,
        )


if __name__ == "__main__":
    unittest.main()
