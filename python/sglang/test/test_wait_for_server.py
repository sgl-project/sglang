from unittest.mock import Mock, patch

import pytest
import requests

from sglang.utils import wait_for_server


def test_wait_for_server_early_exit_when_process_dies():
    process = Mock()
    process.poll.return_value = 7

    with patch("sglang.utils.requests.get") as mock_get:
        with pytest.raises(RuntimeError, match="Server process exited with code 7"):
            wait_for_server("http://127.0.0.1:8000", timeout=30, process=process)

    mock_get.assert_not_called()


def test_wait_for_server_keeps_legacy_timeout_without_process():
    with patch(
        "sglang.utils.requests.get",
        side_effect=requests.exceptions.RequestException("connection error"),
    ) as mock_get:
        with patch("sglang.utils.time.sleep", return_value=None):
            with patch("sglang.utils.time.perf_counter", side_effect=[0.0, 2.0]):
                with pytest.raises(
                    TimeoutError,
                    match="Endpoint http://127.0.0.1:8000/v1/models did not become ready within timeout period",
                ):
                    wait_for_server("http://127.0.0.1:8000", timeout=1)

    assert mock_get.call_count == 1
