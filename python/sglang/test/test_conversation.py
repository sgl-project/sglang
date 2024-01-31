from sglang.srt.conversation import generate_chat_conv
from sglang.srt.managers.openai_protocol import (
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
)


def test_chat_completion_to_conv_image():
    """Test that we can convert a chat image request to a convo"""
    request = ChatCompletionRequest(
        model="default",
        messages=[
            ChatCompletionMessageGenericParam(
                role="system", content="You are a helpful AI assistant"
            ),
            ChatCompletionMessageUserParam(
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
            ),
        ],
    )
    conv = generate_chat_conv(request, "vicuna_v1.1")
    assert conv.messages == [
        ["USER", "Describe this image<image>"],
        ["ASSISTANT", None],
    ]
    assert conv.system_message == "You are a helpful AI assistant"
    assert conv.image_data == ["https://someurl.com"]


if __name__ == "__main__":
    test_chat_completion_to_conv_image()
