import openai
client = openai.Client(api_key="EMPTY", base_url="http://127.0.0.1:30000/v1")
import sys


request_1 = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/mixtral_8x7b.jpg"
                    },
                },
                {"type": "text", "text": "Please describe this image. Please list the benchmarks and the models."},
            ],
        },
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True,
)
response_1 = ""


for chunk in request_1:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        response_1 += content
        sys.stdout.write(content)
        sys.stdout.flush()

print()  # Add a newline at the end of the stream

# from decord import VideoReader, cpu
# import numpy as np
# video_path = "/mnt/bn/vl-research/workspace/boli01/projects/demos/sglang_codebase/assets/jobs.mp4"
# max_frames_num = 32
# vr = VideoReader(video_path, ctx=cpu(0))
# total_frame_num = len(vr)
# uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
# frame_idx = uniform_sampled_frames.tolist()
# frames = vr.get_batch(frame_idx).asnumpy()
# # to pil and then base64
# import io
# from PIL import Image
# import base64
# base64_frames = []
# for frame in frames:
#     pil_img = Image.fromarray(frame)
#     buff = io.BytesIO()
#     pil_img.save(buff, format="JPEG")
#     base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
#     base64_frames.append(base64_str)


# messages=[
#     {
#         "role": "user",
#         "content": [
#         ],
#     },
# ]
# # frame format
# frame_format = {
#     "type": "image_url",
#     "image_url": {
#         "url": "data:image/jpeg;base64,{}"
#     },
# }
# for base64_frame in base64_frames:
#     frame_format["image_url"]["url"] = "data:image/jpeg;base64,{}".format(base64_frame)
#     messages[0]["content"].append(frame_format)
# prompt =             {"type": "text", "text": "What is the setting of the video?"}
# messages[0]["content"].append(prompt)

# request_3 = client.chat.completions.create(
#     model="default",
#     messages=messages,
#     temperature=0.7,
#     max_tokens=1024,
#     stream=True,
# )
# print("-"*30)
# response_3 = ""

# for chunk in request_3:
#     if chunk.choices[0].delta.content is not None:
#         content = chunk.choices[0].delta.content
#         response_3 += content
#         sys.stdout.write(content)
#         sys.stdout.flush()