"""
Usage:
pip install opencv-python-headless

python3 srt_example_llava_v.py
"""

import argparse
import csv
import os
import time

import requests

import sglang as sgl


@sgl.function
def video_qa(s, num_frames, video_path, question):
    s += sgl.user(sgl.video(video_path, num_frames) + question)
    s += sgl.assistant(sgl.gen("answer"))


def single(path, num_frames=16):
    state = video_qa.run(
        num_frames=num_frames,
        video_path=path,
        question="Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes",
        temperature=0.0,
        max_new_tokens=1024,
    )
    print(state["answer"], "\n")


def split_into_chunks(lst, num_chunks):
    """Split a list into a specified number of chunks."""
    # Calculate the chunk size using integer division. Note that this may drop some items if not evenly divisible.
    chunk_size = len(lst) // num_chunks

    if chunk_size == 0:
        chunk_size = len(lst)
    # Use list comprehension to generate chunks. The last chunk will take any remainder if the list size isn't evenly divisible.
    chunks = [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
    # Ensure we have exactly num_chunks chunks, even if some are empty
    chunks.extend([[] for _ in range(num_chunks - len(chunks))])
    return chunks


def save_batch_results(batch_video_files, states, cur_chunk, batch_idx, save_dir):
    csv_filename = f"{save_dir}/chunk_{cur_chunk}_batch_{batch_idx}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_name", "answer"])
        for video_path, state in zip(batch_video_files, states):
            video_name = os.path.basename(video_path)
            writer.writerow([video_name, state["answer"]])


def compile_and_cleanup_final_results(cur_chunk, num_batches, save_dir):
    final_csv_filename = f"{save_dir}/final_results_chunk_{cur_chunk}.csv"
    with open(final_csv_filename, "w", newline="") as final_csvfile:
        writer = csv.writer(final_csvfile)
        writer.writerow(["video_name", "answer"])
        for batch_idx in range(num_batches):
            batch_csv_filename = f"{save_dir}/chunk_{cur_chunk}_batch_{batch_idx}.csv"
            with open(batch_csv_filename, "r") as batch_csvfile:
                reader = csv.reader(batch_csvfile)
                next(reader)  # Skip header row
                for row in reader:
                    writer.writerow(row)
            os.remove(batch_csv_filename)


def find_video_files(video_dir):
    # Check if the video_dir is actually a file
    if os.path.isfile(video_dir):
        # If it's a file, return it as a single-element list
        return [video_dir]

    # Original logic to find video files in a directory
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, file))
    return video_files


def batch(video_dir, save_dir, cur_chunk, num_chunks, num_frames=16, batch_size=64):
    video_files = find_video_files(video_dir)
    chunked_video_files = split_into_chunks(video_files, num_chunks)[cur_chunk]
    num_batches = 0

    for i in range(0, len(chunked_video_files), batch_size):
        batch_video_files = chunked_video_files[i : i + batch_size]
        print(f"Processing batch of {len(batch_video_files)} video(s)...")

        if not batch_video_files:
            print("No video files found in the specified directory.")
            return

        batch_input = [
            {
                "num_frames": num_frames,
                "video_path": video_path,
                "question": "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.",
            }
            for video_path in batch_video_files
        ]

        start_time = time.time()
        states = video_qa.run_batch(batch_input, max_new_tokens=512, temperature=0.2)
        total_time = time.time() - start_time
        average_time = total_time / len(batch_video_files)
        print(
            f"Number of videos in batch: {len(batch_video_files)}. Average processing time per video: {average_time:.2f} seconds. Total time for this batch: {total_time:.2f} seconds"
        )

        save_batch_results(batch_video_files, states, cur_chunk, num_batches, save_dir)
        num_batches += 1

    compile_and_cleanup_final_results(cur_chunk, num_batches, save_dir)


if __name__ == "__main__":

    url = "https://raw.githubusercontent.com/EvolvingLMMs-Lab/sglang/dev/onevision_local/assets/jobs.mp4"

    cache_dir = os.path.expanduser("~/.cache")
    file_path = os.path.join(cache_dir, "jobs.mp4")

    os.makedirs(cache_dir, exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad responses

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"File downloaded and saved to: {file_path}")
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run video processing with specified port."
    )

    # Add an argument for the port
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="The master port for distributed serving.",
    )
    parser.add_argument(
        "--chunk-idx", type=int, default=0, help="The index of the chunk to process."
    )
    parser.add_argument(
        "--num-chunks", type=int, default=8, help="The number of chunks to process."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./work_dirs/llava_video",
        help="The directory to save the processed video files.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=os.path.expanduser("~/.cache/jobs.mp4"),
        help="The directory or path for the processed video files.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmms-lab/LLaVA-NeXT-Video-7B",
        help="The model path for the video processing.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="The number of frames to process in each video.",
    )
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)

    # Parse the arguments
    args = parser.parse_args()
    cur_port = args.port
    cur_chunk = args.chunk_idx
    num_chunks = args.num_chunks
    num_frames = args.num_frames

    if "34b" in args.model_path.lower():
        tokenizer_path = "liuhaotian/llava-v1.6-34b-tokenizer"
    elif "7b" in args.model_path.lower():
        tokenizer_path = "llava-hf/llava-1.5-7b-hf"
    else:
        print("Invalid model path. Please specify a valid model path.")
        exit()

    model_override_args = {}
    model_override_args["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
    model_override_args["architectures"] = ["LlavaVidForCausalLM"]
    model_override_args["num_frames"] = args.num_frames
    model_override_args["model_type"] = "llava"

    if "34b" in args.model_path.lower():
        model_override_args["image_token_index"] = 64002

    if args.num_frames == 32:
        model_override_args["rope_scaling"] = {"factor": 2.0, "type": "linear"}
        model_override_args["max_sequence_length"] = 4096 * 2
        model_override_args["tokenizer_model_max_length"] = 4096 * 2
    elif args.num_frames < 32:
        pass
    else:
        print(
            "The maximum number of frames to process is 32. Please specify a valid number of frames."
        )
        exit()

    runtime = sgl.Runtime(
        model_path=args.model_path,  # "liuhaotian/llava-v1.6-vicuna-7b",
        tokenizer_path=tokenizer_path,
        port=cur_port,
        additional_ports=[cur_port + 1, cur_port + 2, cur_port + 3, cur_port + 4],
        model_override_args=model_override_args,
        tp_size=1,
    )
    sgl.set_default_backend(runtime)
    print(f"chat template: {runtime.endpoint.chat_template.name}")

    # Run a single request
    print("\n========== single ==========\n")
    root = args.video_dir
    if os.path.isfile(root):
        video_files = [root]
    else:
        video_files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]  # Add more extensions if needed
    start_time = time.time()  # Start time for processing a single video
    for cur_video in video_files[:1]:
        print(cur_video)
        single(cur_video, num_frames)
    end_time = time.time()  # End time for processing a single video
    total_time = end_time - start_time
    average_time = total_time / len(
        video_files
    )  # Calculate the average processing time
    print(f"Average processing time per video: {average_time:.2f} seconds")
    runtime.shutdown()

    # # Run a batch of requests
    # print("\n========== batch ==========\n")
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    # batch(args.video_dir, args.save_dir, cur_chunk, num_chunks, num_frames, num_chunks)
    # runtime.shutdown()
