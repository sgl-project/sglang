import openai

# Configure client to point to the ROUTER port (8902)
client = openai.Client(base_url="http://localhost:8902/v1", api_key="EMPTY")

# created a dummy video with ffmpeg
# ffmpeg -f lavfi -i "testsrc=size=1280x720:rate=30:duration=10" \
#        -pix_fmt yuv420p -c:v libx264 -preset ultrafast \
#        .mp4sample
# Dummy video path
video_url = "/sgl-workspace/sglang/sample.mp4"

try:
    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                    },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)

except openai.BadRequestError as e:
    print("Successfully reproduced the bug!")
    print(f"Error: {e}")
