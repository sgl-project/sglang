import os
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


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


# Override remote URLs with local NPU asset paths so that the upstream
# mixins (which reference these module-level constants) fetch files from
# the local filesystem instead of the network in NPU CI.
import sglang.test.vlm_utils as _upstream
from sglang.test.ascend.test_ascend_utils import (
    AUDIO_BIRD_SONG_PATH,
    AUDIO_TRUMP_WEF_PATH,
    IMAGE_MAN_IRONING_PATH,
    IMAGE_SGL_LOGO_PATH,
    VIDEO_JOBS_PRESENTING_IPOD_PATH,
)

_upstream.IMAGE_SGL_LOGO_URL = IMAGE_SGL_LOGO_PATH
_upstream.IMAGE_MAN_IRONING_URL = IMAGE_MAN_IRONING_PATH
_upstream.AUDIO_BIRD_SONG_URL = AUDIO_BIRD_SONG_PATH
_upstream.AUDIO_TRUMP_SPEECH_URL = AUDIO_TRUMP_WEF_PATH
_upstream.VIDEO_JOBS_URL = VIDEO_JOBS_PRESENTING_IPOD_PATH

_orig_download = _upstream.TestOpenAIMLLMServerBase.get_or_download_file


def _npu_get_or_download_file(self, url: str) -> str:
    if url is None:
        raise ValueError()

    # Local absolute path: use as-is if it exists, otherwise fail fast
    # instead of falling through to the HTTP download branch.
    if os.path.isabs(url):
        if os.path.exists(url):
            return url
        raise FileNotFoundError(f"Local file does not exist: {url}")


_upstream.TestOpenAIMLLMServerBase.get_or_download_file = _npu_get_or_download_file

from sglang.test.vlm_utils import *
