"""The draft factory and the runner build an attention backend for every
speculative forward without a per-backend list of the combinations it serves.

A combination is the forward (multi-step draft decode, draft extend, target
verify), whether the model has sliding-window layers, whether the runner
captures CUDA graphs, and whether its KV storage is translated (the unified
pool, or a draft fused into it). Every backend that can face translated storage
reads its window and its ids through the same window reads and id spaces, so
the combinations once refused there build: a FlashInfer target verify and
graph draft extend with window layers, a trtllm_mha draft extend with window
layers, an MLA target verify. A backend that cannot read translated storage at
all refuses it at its own init, by the storage's views (AITER; its tests are in
`test_fused_draft_window_metadata.py`). Constructors are stood in for: what is
checked is that neither the factory nor the runner refuses.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import hybrid_attn_backend  # noqa: E402
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

_MHA = (
    (TritonAttnBackend, "triton"),
    (FlashInferAttnBackend, "flashinfer"),
    (TRTLLMHAAttnBackend, "trtllm_mha"),
    (FlashAttentionBackend, "fa3"),
)
_MLA = (FlashInferMLAAttnBackend, TRTLLMMLABackend, CuteDslMLABackend)


def _runner(
    *,
    draft: bool = False,
    algorithm: SpeculativeAlgorithm = SpeculativeAlgorithm.EAGLE,
    **extra,
):
    """A runner with sliding-window layers on translated storage."""
    return SimpleNamespace(
        sliding_window_size=128,
        kv_index_translator=SimpleNamespace(is_translating=True),
        is_draft_worker=draft,
        spec_algorithm=algorithm,
        draft_attention_backend=None,
        **extra,
    )


def _built(cls):
    """An instance of ``cls`` as its constructor would leave its class."""
    return object.__new__(cls)


class _Container:
    """A multi-step draft backend: one step backend per draft step."""

    def __init__(self, cls, steps=2):
        self.attn_backends = [_built(cls) for _ in range(steps)]


class TestDraftFactory(unittest.TestCase):
    """The draft factory checks what it builds: every step of a multi-step
    draft decode, and the draft-extend backend."""

    def _decode(self, cls, name):
        runner = _runner(draft=True)
        container = _Container(cls)
        factory = draft_utils.DraftBackendFactory(
            runner, topk=1, speculative_num_steps=3
        )
        with (
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

    def _extend(self, cls, name):
        runner = _runner(draft=True)
        factory = draft_utils.DraftBackendFactory(
            runner, topk=1, speculative_num_steps=3
        )
        with (
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

    def test_window_drafts_build_on_every_backend(self):
        """Multi-step draft decode and draft extend, which the lists once
        refused on translated storage for some backends, now build."""
        for cls, name in _MHA:
            with self.subTest(backend=name, mode="draft decode"):
                built = self._decode(cls, name)
                self.assertEqual(len(built.attn_backends), 2)
            with self.subTest(backend=name, mode="draft extend"):
                self.assertIsInstance(self._extend(cls, name), cls)


class TestRunnerBackend(unittest.TestCase):
    """A speculative runner builds its own backend for target verify (the
    target) or draft extend (a draft) without consulting a combination list."""

    def _build(
        self,
        prefill_cls,
        decode_cls,
        *,
        draft=False,
        mode="prefill",
        algorithm=SpeculativeAlgorithm.EAGLE,
    ):
        runner = _runner(
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

    def test_window_target_verify_builds_on_every_backend(self):
        """A translated target verify with window layers: once refused on
        FlashInfer and the MLA family, it now builds, alone or as the verifying
        child of a split pair."""
        for cls, _ in _MHA:
            with self.subTest(backend=cls.__name__):
                self.assertIsInstance(self._build(cls, cls), cls)
        for cls in _MLA:
            with self.subTest(backend=cls.__name__):
                self.assertIsInstance(self._build(cls, cls), cls)
        for mode in ("prefill", "decode"):
            with self.subTest(split_pair_verifies_on=mode):
                self._build(TritonAttnBackend, FlashInferMLAAttnBackend, mode=mode)

    def test_a_draft_runner_builds_its_draft_extend_backend(self):
        for cls, _ in _MHA:
            with self.subTest(backend=cls.__name__):
                self.assertIsInstance(self._build(cls, cls, draft=True), cls)


if __name__ == "__main__":
    unittest.main()
