import os
import warnings
from types import SimpleNamespace

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
    """Override multimodal asset URLs/paths to local NPU files.

    NPU CI uses local asset files instead of downloading from the remote
    sgl-test-files mirror. Overriding the class attributes defined in
    `sglang.test.vlm_utils.TestOpenAIMLLMServerBase` here keeps the test
    mixins reusable across backends.
    """

    image_man_ironing_url = IMAGE_MAN_IRONING_PATH
    image_sgl_logo_url = IMAGE_SGL_LOGO_PATH
    video_jobs_url = VIDEO_JOBS_PRESENTING_IPOD_PATH
    audio_trump_speech_url = AUDIO_TRUMP_WEF_PATH
    audio_bird_song_url = AUDIO_BIRD_SONG_PATH
    video_max_frames_num = 4


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
