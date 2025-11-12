# client.py
import asyncio
import base64
import io

import aiohttp
from PIL import Image

IMAGE_TOKEN_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"


def create_image_bytes(color, width=512, height=512, format="PNG"):
    """Create an image of the specified color and return its bytes as base64 string."""
    # Create image with specified color
    img = Image.new('RGB', (width, height), color=color)
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    
    # Return base64 encoded string
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def one_image_request(image_data):
    # print(f"Processing image {image_data}")
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://0.0.0.0:30000/generate",
            json={
                "text": f"Hello {IMAGE_TOKEN_TEXT}",
                "image_data": image_data,
                "sampling_params": {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "stop_token_ids": [1],  # <|separator|>
                },
            },
        )
        response_json = await response.json()
        return response_json["text"]

async def two_images_request(image_data_1, image_data_2):
    # print(f"Processing image {image_data_1} and {image_data_2}")
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://0.0.0.0:30000/generate",
            json={
                "text": f"Hello {IMAGE_TOKEN_TEXT} {IMAGE_TOKEN_TEXT}",
                "image_data": [image_data_1, image_data_2],
                "sampling_params": {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "stop_token_ids": [1],  # <|separator|>
                },
            },
        )
        response_json = await response.json()
        return response_json["text"]

async def main():
    # Create black and white images as base64-encoded bytes
    black_image_bytes = create_image_bytes(color='black')
    white_image_bytes = create_image_bytes(color='white')
    
    print("Created black and white images")
    one_image_response = await one_image_request(black_image_bytes)
    two_image_response = await two_images_request(black_image_bytes, white_image_bytes)
    print(f"One image response: {one_image_response}")
    print(f"Two image response: {two_image_response}")
    


if __name__ == "__main__":
    asyncio.run(main())