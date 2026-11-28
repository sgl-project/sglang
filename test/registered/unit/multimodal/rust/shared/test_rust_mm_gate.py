"""The Rust-MM launch gate's family selection.

``RUST_MM_FAMILIES`` decides which models the Rust pipeline serves;
for everything else ``rust_mm_family_for`` must return ``None``, which
``RustServer.launch`` turns into a hard launch error. Pins that non-Qwen
multimodal models — Inkling being the in-tree case — keep their Python
processor and never match a Rust family, so growing the registry cannot
silently reroute them.
"""

import unittest
from types import SimpleNamespace

from transformers import (
    Glm5NextImageProcessor,
    Qwen2VLImageProcessor,
    Qwen3VLVideoProcessor,
)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.multimodal_processor import (  # noqa: E402
    get_mm_processor_cls,
    import_processors,
)
from sglang.srt.runtime_context import publish
from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RustMmModelDirSpec,
    RustMmNemotronSpec,
    RustMmProcessor,
    rust_mm_family_for,
)
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


def processor_cls_for(architecture, model_type):
    """Through the production selection, as `resolve_spec` calls it."""
    hf_config = SimpleNamespace(architectures=[architecture], model_type=model_type)
    # `model_impl` is read from the bags now, so it has to be published rather
    # than handed over on a stand-in.
    publish(ServerArgs(model_path="dummy", model_impl="sglang"), role="tokenizer")
    return get_mm_processor_cls(hf_config)


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

    def test_qwen3_vl_video_rule_launches_with_video(self):
        """Every model that inherits Qwen3-VL's video rule, with HF's default
        processors and no `--mm-process-config`, must keep video on; the gate
        otherwise drops to image-only with only a warning. Qwen2.5-VL places
        video differently and must stay image-only."""
        publish(ServerArgs(model_path="dummy", model_impl="sglang"), role="tokenizer")
        cases = (
            ("Qwen3VLForConditionalGeneration", "qwen3_vl", True),
            ("Qwen3VLMoeForConditionalGeneration", "qwen3_vl_moe", True),
            ("Qwen3_5ForConditionalGeneration", "qwen3_5", True),
            ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe", True),
            ("Qwen4ExpForConditionalGeneration", "qwen4_exp", True),
            ("Qwen2_5_VLForConditionalGeneration", "qwen2_5_vl", False),
        )
        for architecture, model_type, video in cases:
            with self.subTest(model_type=model_type):
                host = RustMmProcessor.__new__(RustMmProcessor)
                host.model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=[architecture],
                        model_type=model_type,
                        image_token_id=1,
                        video_token_id=2,
                        vision_start_token_id=3,
                        vision_end_token_id=4,
                    )
                )
                host._processor = SimpleNamespace(
                    image_processor=Qwen2VLImageProcessor(patch_size=16),
                    video_processor=Qwen3VLVideoProcessor(),
                )
                spec = host.resolve_spec()
                self.assertIsNotNone(spec)
                self.assertEqual(spec.video is not None, video)
                if video:
                    self.assertEqual(spec.video.video_token_id, 2)

    def glm5_next_host(self, tokenizer_path="dummy", **server_args):
        publish(
            ServerArgs(
                model_path="dummy",
                tokenizer_path=tokenizer_path,
                model_impl="sglang",
                **server_args,
            ),
            role="tokenizer",
        )
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(
            model_path="zai-org/GLM-5.3-Flash",
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"],
                model_type="glm5_next",
                image_token_id=1,
                video_token_id=2,
            ),
        )
        host.server_args = SimpleNamespace(revision="r1")
        host._processor = SimpleNamespace(image_processor=Glm5NextImageProcessor())
        return host

    def test_glm5_next_spec_comes_from_the_checkpoint(self):
        """Glm4vImageProcessor reports the image token as the video token too: the
        drain must hand the scheduler that id, not hf_config.video_token_id."""
        spec = self.glm5_next_host().resolve_spec()
        self.assertEqual(
            spec,
            RustMmModelDirSpec(
                family="glm5_next",
                feature_shm=False,
                model_path="zai-org/GLM-5.3-Flash",
                revision="r1",
                resample="aten_u8",
                image_token_id=1,
                video_token_id=1,
            ),
        )

    def test_glm5_next_refuses_processor_overrides(self):
        """Rust reads the processor settings and config.json next to --model-path,
        so a server override of what Python reads from them, or a processor from
        another path, must fail the launch rather than be ignored."""
        for server_args in (
            {"mm_process_config": {"video": {"fps": 1}}},
            {"mm_process_config": {"image": {"max_pixels": 4096}}},
            {"tokenizer_path": "other"},
            {
                "json_model_override_args": '{"vision_config": {"spatial_merge_size": 1}}'
            },
            {"json_model_override_args": '{"image_token_id": 5}'},
        ):
            host = self.glm5_next_host(**server_args)
            self.assertIsNone(host.resolve_spec(), server_args)
        host = self.glm5_next_host(json_model_override_args='{"rope_theta": 1.0}')
        self.assertIsNotNone(host.resolve_spec())

    def nemotron_host(self, context_length, **server_args):
        publish(
            ServerArgs(
                model_path="dummy",
                tokenizer_path="dummy",
                model_impl="sglang",
                context_length=context_length,
                **server_args,
            ),
            role="tokenizer",
        )
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(
            model_path="nvidia/Nemotron-VL",
            hf_config=SimpleNamespace(
                architectures=["NemotronH_Omni_Reasoning_V3"],
                model_type="nemotron_h_omni",
                img_context_token="<image>",
                img_start_token="<img>",
                img_end_token="</img>",
                # The checkpoint's preprocessor_config.json says 1024; SGLang reads
                # vision_config, which has none.
                video_target_num_patches=0,
                video_maintain_aspect_ratio=True,
                dynamic_resolution=True,
                video_pruning_rate=0.7,
                patch_size=16,
                downsample_ratio=0.5,
            ),
        )
        host.server_args = SimpleNamespace(revision="r1")
        ids = {"<image>": 18, "<img>": 19, "</img>": 20}
        host._processor = SimpleNamespace(
            image_processor=type("NemotronH_Omni_Reasoning_V3ImageProcessor", (), {})(),
            tokenizer=SimpleNamespace(convert_tokens_to_ids=ids.__getitem__),
        )
        return host

    def test_nemotron_tile_path_takes_no_rust_pipeline(self):
        """``dynamic_resolution`` off (a ``--json-model-override-args`` override)
        selects Python's tile path, which Rust does not serve; the checkpoint
        files Rust reads still say dynamic, so Rust would build other features."""
        host = self.nemotron_host(None)
        host.model_config.hf_config.dynamic_resolution = False
        self.assertIsNone(host.resolve_spec())

    def test_nemotron_refuses_processor_overrides(self):
        """NanoNemotronVLImageProcessor reads these from hf_config and Rust from
        config.json; an override of a value Rust takes from Python still launches."""
        for override in (
            '{"patch_size": 14}',
            '{"norm_mean": [0, 0, 0]}',
            '{"max_num_patches": 1024}',
            '{"video_context_token": "<v>"}',
        ):
            host = self.nemotron_host(None, json_model_override_args=override)
            self.assertIsNone(host.resolve_spec(), override)
        host = self.nemotron_host(
            None, json_model_override_args='{"video_pruning_rate": 0.0}'
        )
        self.assertIsNotNone(host.resolve_spec())

    def test_nemotron_spec_takes_the_python_processor_values(self):
        """NanoNemotronVLImageProcessor hands the scheduler the <image> id as the
        video token too, budgets images by --context-length (8192 unset), and
        sizes video frames by hf_config, not by the processor files Rust reads."""
        for context_length, max_model_len in ((16384, 16384), (None, 8192)):
            spec = self.nemotron_host(context_length).resolve_spec()
            self.assertEqual(
                spec,
                RustMmModelDirSpec(
                    family="nemotron_vl",
                    feature_shm=False,
                    model_path="nvidia/Nemotron-VL",
                    revision="r1",
                    resample="pil",
                    image_token_id=18,
                    video_token_id=18,
                    vision_start_token_id=19,
                    vision_end_token_id=20,
                    nemotron=RustMmNemotronSpec(
                        max_model_len=max_model_len,
                        video_target_num_patches=0,
                        video_maintain_aspect_ratio=True,
                        video_pruning_rate=0.7,
                        patch_size=16,
                        downsample_factor=2,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
