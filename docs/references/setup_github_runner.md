# Set Up Self-Hosted Runners for GitHub Action

## Add a Runner

### Step 1: Start a docker container.

You can mount a folder for the shared huggingface model weights cache. The command below uses `/tmp/huggingface` as an example.

```
docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04
# Nvidia
docker run --shm-size 128g -it -v /tmp/huggingface:/hf_home --gpus all nvidia/cuda:12.1.1-devel-ubuntu22.04 /bin/bash
# AMD
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 128g -it -v /tmp/huggingface:/hf_home lmsysorg/sglang:v0.4.8.post1-rocm630 /bin/bash
# AMD just the last 2 GPUs
docker run --rm --device=/dev/kfd --device=/dev/dri/renderD176 --device=/dev/dri/renderD184 --group-add video --shm-size 128g -it -v /tmp/huggingface:/hf_home lmsysorg/sglang:v0.4.8.post1-rocm630 /bin/bash
```

### Step 2: Configure the runner by `config.sh`

Run these commands inside the container.

```
apt update && apt install -y curl python3-pip git
export RUNNER_ALLOW_RUNASROOT=1
```

Then follow https://github.com/sgl-project/sglang/settings/actions/runners/new?arch=x64&os=linux to run `config.sh`

**Notes**
- Do not need to specify the runner group
- Give it a name (e.g., `test-sgl-gpu-0`) and some labels (e.g., `1-gpu-runner`). The labels can be edited later in Github Settings.
- Do not need to change the work folder.

### Step 3: Run the runner by `run.sh`

- Set up environment variables
```
export HF_HOME=/hf_home
export SGLANG_IS_IN_CI=true
export HF_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
export CUDA_VISIBLE_DEVICES=0
```

- Run it forever
```
while true; do ./run.sh; echo "Restarting..."; sleep 2; done
```
