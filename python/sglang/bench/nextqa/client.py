"""
Launch the benchmark client for Llava-video model.
Sends all the videos in a directory to the server and ask the LLM to discribe.
Example: unpack videos into ./videos and run the following command:
python client.py --port 3000
"""

import argparse
import os
import sys
import time
from typing import List

import requests
from video import NExTQALoader, Video, VideoFileLoader, VideoPrompt

import sglang as sgl
from sglang.utils import encode_video_base64


@sgl.function
def video_qa(s, num_frames, video_path, question):
    # note: the order of video and question does not matter
    s += sgl.user(
        sgl.video(video_path, num_frames) + question
    )  # build request and encode video frames
    # TODO: video_path
    # s += sgl.user(question + sgl.video(video_path, num_frames))
    s += sgl.assistant(sgl.gen("answer"))  # send request to the LLM


# @sgl.function
# def next_qa(s, num_frames, video_path, question, ):


class VideoClient:
    def __init__(self, port: int):
        self.port = port
        # self.port = port
        # self.endpoint = sgl.RuntimeEndpoint(f"http://localhost:{port}")
        # sgl.set_default_backend(self.endpoint)
        # print(f"chat template: {self.endpoint.chat_template.name}")

    def single(self, video_path: str, num_frames):
        print("single() is not implemented yet")

    def batch(self, video_dir: str, num_frames, batch_size, save_dir):
        print("batch() is not implemented yet")


class VideoClientSgl(VideoClient):
    def __init__(self, port: int):
        super().__init__(port)
        self.endpoint = sgl.RuntimeEndpoint(f"http://localhost:{port}")
        sgl.set_default_backend(self.endpoint)
        print(f"chat template: {self.endpoint.chat_template.name}")

    def single(self, video: Video, prompt: str):
        """
        Handle a single video
        """
        if video.num_frames == 0:
            print(f"Video {video.path} has 0 frames. Skipping...")
            return

        print(video)

        start_time = time.time()
        state = video_qa.run(
            num_frames=video.num_frames,
            video_path=video.path,
            question=prompt,
            temperature=0.0,
            max_new_tokens=1024,
        )
        answer = state["answer"]
        total_time = time.time() - start_time

        print("Prompt: ", prompt)
        print(f"Answer: {answer}")
        print(f"Latency: {total_time} seconds.")

    def batch(self, video_prompts: List[VideoPrompt], save_dir="./answers"):
        """
        Handle a batch of videos
        """
        # remove invalid videos
        valid_videos = []
        for video in video_prompts:
            if video.num_frames == 0:
                print(f"Video {video.path} has 0 frames. Skipping...")
            else:
                valid_videos.append(video)
        if len(valid_videos) == 0:
            print("No valid videos in this batch.")
            return
        videos = valid_videos

        # process batch input
        print(f"Processing batch of {len(videos)} video(s)...")

        batch_input = [
            {
                "num_frames": video.num_frames,
                "video_path": video.path,
                "question": video.prompt,
            }
            for video in videos
        ]

        start_time = time.time()

        # query
        states = video_qa.run_batch(batch_input, max_new_tokens=512, temperature=0.2)
        # save batch results
        for state, video in zip(states, videos):
            with open(
                os.path.join(save_dir, os.path.basename(video.path) + ".log"), "w"
            ) as f:
                f.write(state["answer"])

        total_time = time.time() - start_time
        throughput = len(videos) / total_time
        print(
            f"Number of videos in batch: {len(videos)}.\n"
            f"Total time for this batch: {total_time:.2f} seconds.\n"
            f"Throughput: {throughput:.2f} videos/second"
        )


class VideoDiscrptClientSgl(VideoClientSgl):
    """
    SGLang client for Video Discription
    """

    def __init__(self, port: int):
        super().__init__(port)

    def single(self, video: Video):
        super().single(
            video,
            "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.",
        )

    def batch(self, videos: List[Video], save_dir="./answers"):
        prompt = "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."
        videos = [VideoPrompt(video.path, video.num_frames, prompt) for video in videos]
        super().batch(
            video_prompts=videos,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video client connected to specific port."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="The master port for distributed serving.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="The directory or path for the processed video files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=sys.maxsize,
        help="The maximum number of frames to process in each video.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output",
        help="The directory to save the processed video files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Whether to process videos in batch.",
    )

    args = parser.parse_args()

    # -- load files and process
    # client = VideoDiscrptClientSgl(args.port)
    # videos = VideoFileLoader(
    #     video_dir=args.video_dir, batch_size=args.batch_size, max_frames=args.max_frames
    # )
    videos = NExTQALoader(
        video_dir=args.video_dir, max_frames=args.max_frames, batch_size=args.batch_size
    )

    # print(args.max_frames)
    # if args.batch_size > 1:
    #     if not os.path.exists(args.save_dir):
    #         os.makedirs(args.save_dir)
    #     for batch in videos:
    #         client.batch(batch, save_dir=args.save_dir)
    # else:
    #     for video in videos:
    #         client.single(video)

    # -- load NExTQA and process with SGLang frontend
    # client = VideoClientSgl(args.port)
    # if args.batch_size > 1:
    #     for batch in videos:
    #         # TODO: can fail if the frame size (W*H) is too large
    #         client.batch(batch, save_dir=args.save_dir)
    # else:
    #     for video in videos:
    #         client.single(video, video.prompt)

    # -- load NExTQA and process with chat completions APIs
    payload = {
        "model": "lmms-lab/LLaVA-NeXT-Video-7B",
        "temperature": 0.0,
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }

    for video in videos:
        path = video.path
        num_frames = video.num_frames
        base64_data = encode_video_base64(path, num_frames)
        # print(base64_data)
        message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_data}},
                {"type": "text", "text": video.prompt},
            ],
        }
        payload["messages"] = [message]
        payload["max_tokens"] = 1024
        print("send: ", message["content"][1])
        response = requests.post(
            url=f"http://localhost:{args.port}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        print(response.json())

    # -- TODO: load NExTQA and process with /generate APIs
