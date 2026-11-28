from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.lang.backend.vertexai as vertexai_mod
from sglang.lang.backend.anthropic import Anthropic
from sglang.lang.backend.vertexai import VertexAI


def _make_anthropic_backend():
    backend = Anthropic.__new__(Anthropic)
    backend.model_name = "claude-3-5-sonnet"
    backend.client = MagicMock()
    return backend


def test_anthropic_generate_does_not_mutate_stream_executor_messages():
    backend = _make_anthropic_backend()
    backend.client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="hello")]
    )

    original_messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    executor = SimpleNamespace(messages_=list(original_messages), text_="")
    sampling_params = MagicMock()
    sampling_params.to_anthropic_kwargs.return_value = {"max_tokens": 16}

    comp, meta = backend.generate(executor, sampling_params)
    assert comp == "hello"
    assert meta == {}
    # StreamExecutor.messages_ must retain the system message for subsequent turns
    assert executor.messages_ == original_messages
    backend.client.messages.create.assert_called_once_with(
        model="claude-3-5-sonnet",
        system="Be concise.",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=16,
    )


def test_anthropic_generate_stream_does_not_mutate_stream_executor_messages():
    backend = _make_anthropic_backend()
    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = SimpleNamespace(text_stream=["hel", "lo"])
    stream_ctx.__exit__.return_value = False
    backend.client.messages.stream.return_value = stream_ctx

    original_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Question"},
    ]
    executor = SimpleNamespace(messages_=list(original_messages), text_="")
    sampling_params = MagicMock()
    sampling_params.to_anthropic_kwargs.return_value = {"max_tokens": 16}

    chunks = list(backend.generate_stream(executor, sampling_params))
    assert chunks == [("hel", {}), ("lo", {})]
    assert executor.messages_ == original_messages
    backend.client.messages.stream.assert_called_once_with(
        model="claude-3-5-sonnet",
        system="System prompt",
        messages=[{"role": "user", "content": "Question"}],
        max_tokens=16,
    )


def test_vertexai_text_to_vertexai_input_preserves_segment_order():
    backend = VertexAI.__new__(VertexAI)
    backend.chat_template = SimpleNamespace(image_token="<image>")

    fake_image_cls = MagicMock()
    fake_image_cls.from_bytes.side_effect = lambda b: f"IMG({b.decode()})"

    with patch.object(vertexai_mod, "Image", fake_image_cls, create=True):
        text = "first <image><image> middle <image> last"
        images = [
            ("img1.png", b"one"),
            ("img2.png", b"two"),
            ("img3.png", b"three"),
        ]
        result = backend.text_to_vertexai_input(text, images)

    assert result == [
        "first ",
        "IMG(one)",
        "IMG(two)",
        " middle ",
        "IMG(three)",
        " last",
    ]
