from sglang.srt.managers.openai_protocol import (
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
)


def test_chat_completion_request_image():
    """Test that Chat Completion Requests with images can be converted."""

    image_request = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://someurl.com"}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    request = ChatCompletionRequest(**image_request)
    assert len(request.messages) == 2
    assert request.messages[0] == ChatCompletionMessageGenericParam(
        role="system", content="You are a helpful AI assistant"
    )
    assert request.messages[1] == ChatCompletionMessageUserParam(
        role="user",
        content=[
            ChatCompletionMessageContentTextPart(
                type="text", text="Describe this image"
            ),
            ChatCompletionMessageContentImagePart(
                type="image_url",
                image_url=ChatCompletionMessageContentImageURL(
                    url="https://someurl.com"
                ),
            ),
        ],
    )


if __name__ == "__main__":
    test_chat_completion_request_image()
