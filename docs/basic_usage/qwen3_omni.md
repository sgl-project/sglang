# Qwen3-Omni Usage

[Qwen3-Omni](https://huggingface.co/collections/Qwen/qwen3-omni) is a natively end-to-end multilingual omni-modal foundation model. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. SGLang supports Qwen3-Omni-30B-A3B-Thinking and Qwen3-Omni-30B-A3B-Instruct.
## Launch Qwen3-Omni with SGLang

To serve Qwen3-Omni models on 4xH20 GPUs:

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Omni-30B-A3B-Instruct --tp 4
```


## Sending Multimodal Requests (Image, Audio, Video)

When interacting with the `qwen3-omni` model, all types of multimodal inputs are sent to the same `/v1/chat/completions` endpoint. The key difference lies in the `content` list within the `messages` payload.

### Image Input

To send a request containing an image, include both `text` and `image_url` types in the `content` list.

```python
import requests
import json

# API server endpoint
url = "http://localhost:30000/v1/chat/completions"

# Request payload
data = {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct", # Specify the Omni model
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        # Provide a publicly accessible image URL
                        "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
    "temperature": 0.7
}

# Send the POST request
response = requests.post(url, json=data)

# Print the model's response
print(response.text)
```

### Audio Input

For audio input, the format is similar to image and video, using the `audio_url` type. This allows you to ask questions about an audio file, such as performing transcription or summarizing its content.

```python
import requests
import json

# API server endpoint
url = "http://localhost:30000/v1/chat/completions"

# Request payload
data = {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct", # Specify the Omni model
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the English audio into text."},
                {
                    "type": "audio_url",
                    "audio_url": {
                        # Provide a publicly accessible audio file URL
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_en.wav"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
    "temperature": 0.2
}

# Send the POST request
response = requests.post(url, json=data)

# Print the model's response
print(response.text)
```

### Video Input

The structure for sending a video request is consistent with image and audio; simply specify the type as `video_url`.

```python
import requests
import json

# API server endpoint
url = "http://localhost:30000/v1/chat/completions"

# Request payload
data = {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct", # Specify the Omni model
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        # Provide a publicly accessible video file URL
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
    "temperature": 0.7
}

# Send the POST request
response = requests.post(url, json=data)

# Print the model's response
print(response.text)
```


## Important Server Parameters

When launching the model server for multimodal support, you can use the following command-line arguments to tune its performance and behavior:

*   `--mm-max-concurrent-calls <value>`
    *   The max concurrent calls for async mm data processing.

*   `--mm-per-request-timeout <seconds>`
    *   The timeout for each multi-modal request in seconds.

*   `--keep-mm-feature-on-device`
    *   Keep multimodal feature tensors on device after processing to save D2H copy.
