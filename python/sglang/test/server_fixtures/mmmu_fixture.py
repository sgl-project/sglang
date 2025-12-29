import logging
import os
import time

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logger = logging.getLogger(__name__)

# Set default mem_fraction_static to 0.8
DEFAULT_MEM_FRACTION_STATIC = 0.8


class MMMUServerBase(CustomTestCase):
    """Server fixture for MMMU VLM tests.

    This fixture handles server lifecycle for single-model MMMU tests.
    For multi-model tests that need to start/stop servers within test methods,
    use MMMUMultiModelTestBase instead.
    """

    model = None
    base_url = DEFAULT_URL_FOR_TEST
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    other_args: list[str] = []
    mem_fraction_static: float = DEFAULT_MEM_FRACTION_STATIC

    @classmethod
    def setUpClass(cls):
        assert cls.model is not None, "Please set cls.model in subclass"

        # Prepare environment variables
        process_env = os.environ.copy()
        process_env["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"

        # Build server args with MMMU-specific settings
        server_args = [
            "--trust-remote-code",
            "--cuda-graph-max-bs",
            "64",
            "--enable-multimodal",
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            *cls.other_args,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            api_key=cls.api_key,
            other_args=server_args,
            env=process_env,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None and cls.process.poll() is None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error killing process: {e}")
        time.sleep(2)
