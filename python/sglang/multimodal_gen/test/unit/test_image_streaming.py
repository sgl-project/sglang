import unittest

from pydantic import ValidationError

from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _stream_image_response,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
)


class TestImageStreamingProtocol(unittest.TestCase):
    def test_request_accepts_streaming_fields(self):
        request = ImageGenerationsRequest(
            prompt="draw a cat",
            stream=True,
            partial_images=2,
        )

        self.assertTrue(request.stream)
        self.assertEqual(request.partial_images, 2)

    def test_partial_images_must_be_in_openai_range(self):
        with self.assertRaises(ValidationError):
            ImageGenerationsRequest(prompt="draw a cat", partial_images=4)


class TestImageStreamingResponse(unittest.IsolatedAsyncioTestCase):
    async def test_stream_image_response_yields_sse_events(self):
        response = ImageResponse(
            id="test-request-id",
            data=[
                ImageResponseData(
                    b64_json="abc123",
                    revised_prompt="draw a cat",
                )
            ],
        )

        chunks = [chunk async for chunk in _stream_image_response(response)]

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("data: "))
        self.assertIn('"type": "image_generation.partial_image"', chunks[0])
        self.assertIn('"partial_image_index": 0', chunks[0])
        self.assertIn('"b64_json": "abc123"', chunks[0])
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

    async def test_stream_image_response_yields_sse_events_url(self):
        response = ImageResponse(
            id="test-request-id",
            data=[
                ImageResponseData(
                    url="http://example.com/image.png",
                    revised_prompt="draw a cat",
                )
            ],
        )

        chunks = [chunk async for chunk in _stream_image_response(response)]

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("data: "))
        self.assertIn('"type": "image_generation.partial_image"', chunks[0])
        self.assertIn('"partial_image_index": 0', chunks[0])
        self.assertIn('"url": "http://example.com/image.png"', chunks[0])
        self.assertNotIn("b64_json", chunks[0])
        self.assertEqual(chunks[1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
