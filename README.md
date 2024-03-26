To refine and clean up the README you've provided for the SGLang project, I'll focus on improving clarity, organization, and conciseness. This includes providing clear installation instructions, simplifying steps where possible, and ensuring the document is easy to navigate. Here's a revised version:

---

<div align="center">
    <img src="assets/logo.png" alt="SGLang Logo" width="400">
</div>

--------------------------------------------------------------------------------

| [**Blog**](https://lmsys.org/blog/2024-01-17-sglang/) | [**Paper**](https://arxiv.org/abs/2312.07104) |

**SGLang** is a structured generation language tailored for large language models (LLMs), enhancing interaction speed and control by integrating an adaptable frontend language with a high-performance runtime system.

### Key Features
- **Flexible Front-End Language**: Facilitates programming of LLM applications, supporting chained generations, advanced prompts, control flows, multi-modalities, parallelism, and external interactions.
- **High-Performance Runtime with RadixAttention**: Boosts complex LLM program execution through KV cache reuse, continuous batching, and tensor parallelism.

## Getting Started

### Installation

#### SGLang Setup
```sh
cd PATH_TO/sglang_video
pip install --upgrade pip
pip install -e "python[all]"
```


#### Dependency Installation

Install the necessary Python packages:

```sh
pip3 install huggingface_hub hf_transfer outlines==0.0.34 opencv-python-headless
pip3 install git+https://github.com/huggingface/transformers.git@56b64bf1a51e29046bb3f8ca15839ff4d6a92c74
```

#### Downloading Models

Before proceeding, ensure you're in the project's model directory:

```sh
cd PATH_TO/sglang_video/work_dirs/llava_next_video_model
```

Choose the set of instructions based on the number of frames you plan to use for inference:

- **For 32 Frames Inference:**

  Download and set up the models:

  ```sh
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='liuhaotian/llava-v1.6-34b', local_dir='./llava-v1.6-Yi-34b-8k')"
  cp llava-v1.6-Yi-34b-8k_config.json llava-v1.6-Yi-34b-8k/config.json 
  cp llava-v1.6-Yi-34b-8k_generation_config.json llava-v1.6-Yi-34b-8k/generation_config.json
  ```

  Repeat the download and setup process for other models as necessary, adjusting the `repo_id` and `local_dir` accordingly.

- **For Inference with Less Than 32 Frames:**

  You only need to download the models without additional setup steps:

  ```sh
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='liuhaotian/llava-v1.6-34b', local_dir='./llava-v1.6-Yi-34b')"
  ```

  Repeat the download process for other models as necessary, adjusting the `repo_id` and `local_dir` accordingly.

#### Additional Notes

- Make sure to replace `PATH_TO` with the actual path to your `sglang_video` directory.
- Adjust the `repo_id` and `local_dir` as needed based on the specific models you intend to use.


### System Requirements
- CUDA >= 12.1. Specific GPU models may require different versions of the Triton compiler:
  - NVIDIA T4: `pip install "triton>=2.2.0"`
  - NVIDIA V100: Install the [nightly version of Triton](https://triton-lang.org/main/getting-started/installation.html).
- For OpenAI backend usage only: `pip install "sglang[openai]"`

### Quick Start Guide
Configure your environment and run examples using the following steps, adjusting paths and parameters as necessary.

1. **Prepare Environment**:
   ```sh
   cd PATH_TO/sglang_video
   ```

2. **Launch and Run on (K) Nodes**:
   - First node:
     ```sh
     bash examples/quick_start/srt_example_llava_v.sh K 0 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO JPEG
     ```
   - Second node:
     ```sh
     bash examples/quick_start/srt_example_llava_v.sh K 1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO JPEG
     ```
   - The K node:
     ```sh
     bash examples/quick_start/srt_example_llava_v.sh K K-1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO JPEG
     ```

Replace `K`, `YOUR_VIDEO_PATH`, `YOUR_MODEL_PATH`, and `FRAMES_PER_VIDEO` with your specific details. Frames are encoded in the either PNG or JPEG format.