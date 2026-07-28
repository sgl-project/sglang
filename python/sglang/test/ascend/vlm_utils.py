import os
import warnings
from types import SimpleNamespace

import sglang.test.vlm_utils as _vlm_utils
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    AUDIO_BIRD_SONG_PATH,
    AUDIO_TRUMP_WEF_PATH,
    IMAGE_MAN_IRONING_PATH,
    IMAGE_SGL_LOGO_PATH,
    VIDEO_JOBS_PRESENTING_IPOD_PATH,
)
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.vlm_utils import AudioOpenAITestMixin as _AudioOpenAITestMixin
from sglang.test.vlm_utils import ImageOpenAITestMixin as _ImageOpenAITestMixin
from sglang.test.vlm_utils import OmniOpenAITestMixin as _OmniOpenAITestMixin
from sglang.test.vlm_utils import TestOpenAIMLLMServerBase as _TestOpenAIMLLMServerBase
from sglang.test.vlm_utils import VideoOpenAITestMixin as _VideoOpenAITestMixin

# Override remote URLs with local NPU asset paths so that the upstream
# mixins (which reference these module-level constants) fetch files from
# the local filesystem instead of the network in NPU CI.
_vlm_utils.IMAGE_MAN_IRONING_URL = IMAGE_MAN_IRONING_PATH
_vlm_utils.IMAGE_SGL_LOGO_URL = IMAGE_SGL_LOGO_PATH
_vlm_utils.VIDEO_JOBS_URL = VIDEO_JOBS_PRESENTING_IPOD_PATH
_vlm_utils.AUDIO_TRUMP_SPEECH_URL = AUDIO_TRUMP_WEF_PATH
_vlm_utils.AUDIO_BIRD_SONG_URL = AUDIO_BIRD_SONG_PATH


class TestVLMModels(CustomTestCase):
    model = ""
    mmmu_accuracy = 0.00
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.35,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        4,
    ]
    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    max_tokens = 30

    @classmethod
    def setUpClass(cls):
        # Removed argument parsing from here
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.base_url}/v1"

        # Prepare environment variables
        process_env = os.environ.copy()

        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=cls.timeout_for_server_launch,
            api_key=cls.api_key,
            other_args=cls.other_args,
            env=process_env,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            print(f"Cleaning up server process {cls.process.pid}")
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                print(f"Error killing server process: {e}")

    def _run_vlm_mmmu_test(self, test_name=""):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=100,
            num_threads=64,
            max_tokens=self.max_tokens,
            return_latency=True,
        )

        metrics, latency = run_eval(args)

        metrics["score"] = round(metrics["score"], 4)
        metrics["latency"] = round(latency, 4)

        print(
            f"\n{'=' * 42}\n"
            f"{self.model} - metrics={metrics} score={metrics['score']}\n"
            f"{'=' * 42}\n"
        )

        self.assertGreaterEqual(
            metrics["score"],
            self.mmmu_accuracy,
            f"Model {self.model} accuracy ({metrics['score']}) "
            f"below expected threshold ({self.mmmu_accuracy:.4f}){test_name}",
        )


class _NPUAssetMixin:
    """NPU-specific overrides for multimodal test mixins.

    1. `get_or_download_file`: support local absolute paths (skip download).
    2. `prepare_video_images_messages`: sample fewer frames (4) to fit NPU memory.
    """

    # Number of frames sampled from video for `prepare_video_images_messages`.
    # NPU has tighter memory budget; 4 frames are sufficient for the test.
    _npu_video_max_frames_num = 4

    def get_or_download_file(self, url: str) -> str:
        if url is None:
            raise ValueError()

        # Local absolute path: use as-is if it exists, otherwise fail fast
        # instead of falling through to the HTTP download branch.
        if os.path.isabs(url):
            if os.path.exists(url):
                return url
            raise FileNotFoundError(f"Local file does not exist: {url}")

        # Fall back to the default remote-download behaviour.
        return super().get_or_download_file(url)

    def prepare_video_images_messages(self, video_path):
        # Replicates the upstream implementation with a smaller
        # `max_frames_num` to fit NPU memory budget.
        import io

        import numpy as np
        import pybase64
        from PIL import Image

        from sglang.srt.utils.video_decoder import VideoDecoderWrapper

        max_frames_num = self._npu_video_max_frames_num
        decoder = VideoDecoderWrapper(video_path)
        total_frame_num = len(decoder)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        frames = decoder.get_frames_at(frame_idx)

        base64_frames = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            buff = io.BytesIO()
            pil_img.save(buff, format="JPEG")
            base64_str = pybase64.b64encode(buff.getvalue()).decode("utf-8")
            base64_frames.append(base64_str)

        messages = [{"role": "user", "content": []}]
        frame_format = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{}"},
            "modalities": "image",
        }

        for base64_frame in base64_frames:
            frame_format["image_url"]["url"] = "data:image/jpeg;base64,{}".format(
                base64_frame
            )
            messages[0]["content"].append(frame_format.copy())

        prompt = {"type": "text", "text": "Please describe the video in detail."}
        messages[0]["content"].append(prompt)

        return messages


class TestOpenAIMLLMServerBase(_NPUAssetMixin, _TestOpenAIMLLMServerBase):
    pass


class AudioOpenAITestMixin(_NPUAssetMixin, _AudioOpenAITestMixin):
    pass


class ImageOpenAITestMixin(_NPUAssetMixin, _ImageOpenAITestMixin):
    pass


class VideoOpenAITestMixin(_NPUAssetMixin, _VideoOpenAITestMixin):
    pass


class OmniOpenAITestMixin(_NPUAssetMixin, _OmniOpenAITestMixin):
    pass
