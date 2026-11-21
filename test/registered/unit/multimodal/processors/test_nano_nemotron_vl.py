import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_MISSING = object()


@contextmanager
def patched_modules(modules):
    old_modules = {name: sys.modules.get(name, _MISSING) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old_module in old_modules.items():
            if old_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def load_processor_module(sample_video_frames):
    numpy = types.ModuleType("numpy")
    numpy.ndarray = type("ndarray", (), {})

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.bfloat16 = object()

    pil = package("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.Image = type("Image", (), {})
    pil.Image = pil_image

    nano_config = types.ModuleType("sglang.srt.configs.nano_nemotron_vl")
    nano_config.NemotronH_Nano_Omni_Reasoning_V3_Config = type(
        "NemotronH_Nano_Omni_Reasoning_V3_Config", (), {}
    )
    nano_config.NemotronH_Nano_VL_V2_Config = type(
        "NemotronH_Nano_VL_V2_Config", (), {}
    )

    schedule_batch = types.ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.Modality = type(
        "Modality", (), {"IMAGE": "image", "VIDEO": "video", "AUDIO": "audio"}
    )
    schedule_batch.MultimodalDataItem = type("MultimodalDataItem", (), {})
    schedule_batch.MultimodalProcessorOutput = type("MultimodalProcessorOutput", (), {})

    nano_models = types.ModuleType("sglang.srt.models.nano_nemotron_vl")
    nano_models.NemotronH_Nano_Omni_Reasoning_V3 = type(
        "NemotronH_Nano_Omni_Reasoning_V3", (), {}
    )
    nano_models.NemotronH_Nano_VL_V2 = type("NemotronH_Nano_VL_V2", (), {})

    parakeet = types.ModuleType("sglang.srt.models.parakeet")
    parakeet.ParakeetExtractor = type("ParakeetExtractor", (), {})

    audio_from_video = types.ModuleType("sglang.srt.multimodal.audio_from_video")
    audio_from_video.extract_audio_from_video_bytes = lambda *args, **kwargs: None

    evs = types.ModuleType("sglang.srt.multimodal.evs")
    evs.EVSProcessor = type("EVSProcessor", (), {})

    internvl_utils = types.ModuleType("sglang.srt.multimodal.internvl_utils")
    for name in (
        "compute_budgeted_image_sizes",
        "get_video_target_size_and_feature_size",
        "image_to_pixel_values",
        "resize_image_to_pixels",
        "video_to_pixel_values",
    ):
        setattr(internvl_utils, name, lambda *args, **kwargs: None)

    base_processor = types.ModuleType("sglang.srt.multimodal.processors.base_processor")
    base_processor.BaseMultimodalProcessor = type("BaseMultimodalProcessor", (), {})
    base_processor.MultimodalSpecialTokens = type("MultimodalSpecialTokens", (), {})

    common = types.ModuleType("sglang.srt.utils.common")
    common.sample_video_frames = sample_video_frames

    repo_root = Path(__file__).parents[5]
    module_path = (
        repo_root / "python/sglang/srt/multimodal/processors/nano_nemotron_vl.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_nano_nemotron_vl_processor", module_path
    )
    module = importlib.util.module_from_spec(spec)

    with patched_modules(
        {
            "numpy": numpy,
            "torch": torch,
            "PIL": pil,
            "PIL.Image": pil_image,
            "sglang": package("sglang"),
            "sglang.srt": package("sglang.srt"),
            "sglang.srt.configs": package("sglang.srt.configs"),
            "sglang.srt.configs.nano_nemotron_vl": nano_config,
            "sglang.srt.managers": package("sglang.srt.managers"),
            "sglang.srt.managers.schedule_batch": schedule_batch,
            "sglang.srt.models": package("sglang.srt.models"),
            "sglang.srt.models.nano_nemotron_vl": nano_models,
            "sglang.srt.models.parakeet": parakeet,
            "sglang.srt.multimodal": package("sglang.srt.multimodal"),
            "sglang.srt.multimodal.audio_from_video": audio_from_video,
            "sglang.srt.multimodal.evs": evs,
            "sglang.srt.multimodal.internvl_utils": internvl_utils,
            "sglang.srt.multimodal.processors": package(
                "sglang.srt.multimodal.processors"
            ),
            "sglang.srt.multimodal.processors.base_processor": base_processor,
            "sglang.srt.utils": package("sglang.srt.utils"),
            "sglang.srt.utils.common": common,
        }
    ):
        spec.loader.exec_module(module)

    return module


class FakeVideo:
    avg_fps = 2

    def get_frames_at(self, frames):
        return [f"frame-{frame}" for frame in frames]


class TestNanoNemotronVLVideoParsing(unittest.TestCase):
    def test_parse_video_preserves_static_default_call(self):
        calls = []

        def sample_video_frames(video, *, desired_fps, max_frames):
            calls.append((video, desired_fps, max_frames))
            return [0, 2]

        module = load_processor_module(sample_video_frames)
        video = FakeVideo()

        video_array, timestamps = module.NanoNemotronVLImageProcessor.parse_video(video)

        self.assertEqual(calls, [(video, module.DESIRED_FPS, module.MAX_FRAMES)])
        self.assertEqual(video_array, ["frame-0", "frame-2"])
        self.assertEqual(timestamps, [0.0, 1.0])

    def test_parse_video_accepts_configured_sampling_values(self):
        calls = []

        def sample_video_frames(video, *, desired_fps, max_frames):
            calls.append((desired_fps, max_frames))
            return [1]

        module = load_processor_module(sample_video_frames)

        video_array, timestamps = module.NanoNemotronVLImageProcessor.parse_video(
            FakeVideo(), desired_fps=7, max_frames=9
        )

        self.assertEqual(calls, [(7, 9)])
        self.assertEqual(video_array, ["frame-1"])
        self.assertEqual(timestamps, [0.5])


if __name__ == "__main__":
    unittest.main()
