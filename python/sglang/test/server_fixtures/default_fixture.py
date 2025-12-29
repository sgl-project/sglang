import logging
import time
from contextlib import contextmanager

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logger = logging.getLogger(__name__)


@contextmanager
def openai_api_env(api_key: str):
    """Context manager to set OpenAI API environment variables."""
    import os

    original_api_key = os.environ.get("OPENAI_API_KEY")
    original_base_url = os.environ.get("OPENAI_API_BASE")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = f"{DEFAULT_URL_FOR_TEST}/v1"

    try:
        yield
    finally:
        if original_api_key is not None:
            os.environ["OPENAI_API_KEY"] = original_api_key
        else:
            del os.environ["OPENAI_API_KEY"]

        if original_base_url is not None:
            os.environ["OPENAI_API_BASE"] = original_base_url
        else:
            del os.environ["OPENAI_API_BASE"]


class DefaultServerBase(CustomTestCase):
    model = None
    base_url = DEFAULT_URL_FOR_TEST
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    other_args: list[str] = []

    # For OpenAI API settings
    api_key = "sk-123456"

    @classmethod
    def setUpClass(cls):
        assert cls.model is not None, "Please set cls.model in subclass"

        # Set OpenAI API key and base URL environment variables.
        # Needed for lmm-evals to work.
        with openai_api_env(cls.api_key):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=cls.other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(2)
