"""The Rust-MM launch gate's family selection.

``RUST_MM_FAMILIES`` decides which models the Rust pipeline serves;
for everything else ``rust_mm_family_for`` must return ``None``, which
``RustServer.launch`` turns into a hard launch error. Pins that non-Qwen
multimodal models — Inkling being the in-tree case — keep their Python
processor and never match a Rust family, so growing the registry cannot
silently reroute them.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.multimodal_processor import (  # noqa: E402
    get_mm_processor_cls,
    import_processors,
)
from sglang.srt.runtime_context import publish  # noqa: E402
from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.multimodal import rust_mm_family_for  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


def processor_cls_for(architecture, model_type):
    """Through the production selection, as `resolve_spec` calls it."""
    hf_config = SimpleNamespace(architectures=[architecture], model_type=model_type)
    # `model_impl` is read from the bags now, so it has to be published rather
    # than handed over on a stand-in.
    publish(ServerArgs(model_path="dummy", model_impl="sglang"), role="tokenizer")
    return get_mm_processor_cls(hf_config)


def _all1_parallel(**overrides):
    parallel = SimpleNamespace(
        nnodes=1,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        attn_dp_enabled=False,
    )
    parallel.__dict__.update(overrides)
    return parallel


@contextmanager
def _resolve_multimodal_patches(*, root, parallel, disaggregation_mode, processor_ctor):
    with (
        patch.object(
            server_module.envs.SGLANG_DECODED_MEDIA_SHM_ROOT,
            "get",
            return_value=root,
        ),
        patch.object(server_module, "get_parallel", return_value=parallel),
        patch.object(
            server_module,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=disaggregation_mode),
        ),
        patch.object(server_module, "RustMmProcessor", processor_ctor),
    ):
        yield


class TestRustMmGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import_processors("sglang.srt.multimodal.processors")

    def test_qwen_vl_resolves_its_family(self):
        cls = processor_cls_for("Qwen2_5_VLForConditionalGeneration", "qwen2_5_vl")
        family = rust_mm_family_for(cls, "qwen2_5_vl")
        self.assertEqual(family and family.name, "qwen_vl")

    def test_inkling_keeps_its_python_processor(self):
        from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor

        cls = processor_cls_for("InklingForConditionalGeneration", "inkling_model")
        self.assertIs(cls, InklingMultimodalProcessor)
        self.assertIsNone(rust_mm_family_for(cls, "inkling_model"))

    def test_family_requires_both_processor_and_model_type(self):
        qwen = processor_cls_for("Qwen2_5_VLForConditionalGeneration", "qwen2_5_vl")
        self.assertIsNone(rust_mm_family_for(qwen, "inkling_model"))
        # Identity, not name: an override class must not match (the
        # SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE contract).
        impostor = type("QwenVLImageProcessor", (), {})
        self.assertIsNone(rust_mm_family_for(impostor, "qwen2_5_vl"))

    def test_decoded_root_allows_null_prefill_decode_at_per_engine_all1(self):
        scheduler = SimpleNamespace(
            server_args=sentinel.server_args,
            model_config=sentinel.model_config,
            processor=sentinel.processor,
        )
        for mode in ("null", "prefill", "decode"):
            processor = Mock()
            processor.resolve_spec.return_value = sentinel.mm_spec
            with _resolve_multimodal_patches(
                root="/decoded-root",
                parallel=_all1_parallel(),
                disaggregation_mode=mode,
                processor_ctor=Mock(return_value=processor),
            ):
                self.assertIs(
                    RustServer._resolve_multimodal(scheduler), sentinel.mm_spec
                )
            processor.resolve_spec.assert_called_once_with()

    def test_decoded_root_rejects_unsupported_parallelism_before_processor(self):
        scheduler = SimpleNamespace(
            server_args=sentinel.server_args,
            model_config=sentinel.model_config,
            processor=sentinel.processor,
        )
        bad_parallels = (
            _all1_parallel(nnodes=2),
            _all1_parallel(tp_size=2),
            _all1_parallel(dp_size=2),
            _all1_parallel(pp_size=2),
            _all1_parallel(attn_dp_enabled=True),
        )
        for mode in ("null", "prefill", "decode"):
            for parallel in bad_parallels:
                ctor = Mock(
                    side_effect=AssertionError(
                        "RustMmProcessor must not be constructed"
                    )
                )
                with _resolve_multimodal_patches(
                    root="/decoded-root",
                    parallel=parallel,
                    disaggregation_mode=mode,
                    processor_ctor=ctor,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "single-node single-rank serving per engine",
                    ):
                        RustServer._resolve_multimodal(scheduler)
                ctor.assert_not_called()

    def test_decoded_root_disabled_does_not_constrain_encoded_pd(self):
        scheduler = SimpleNamespace(
            server_args=sentinel.server_args,
            model_config=sentinel.model_config,
            processor=sentinel.processor,
        )
        for mode in ("prefill", "decode"):
            processor = Mock()
            processor.resolve_spec.return_value = sentinel.mm_spec
            with _resolve_multimodal_patches(
                root="",
                # Encoded P/D may use multi-rank engines; the decoded root gate
                # must not fire when the root env is unset.
                parallel=_all1_parallel(tp_size=2, attn_dp_enabled=True),
                disaggregation_mode=mode,
                processor_ctor=Mock(return_value=processor),
            ):
                self.assertIs(
                    RustServer._resolve_multimodal(scheduler), sentinel.mm_spec
                )


if __name__ == "__main__":
    unittest.main()
