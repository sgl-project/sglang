# Set Up Self-hosted Runners for GitHub Action

## Config Runner

1. Start a docker container.

You can mount a folder for the shared huggingface cache. The command below uses `/tmp/huggingface` as an example.

```
docker pull nvidia/cuda:12.1.1-devel-ubuntu20.04
docker run --shm-size 64g -it -v /tmp/huggingface/:/hf_home --gpus all nvidia/cuda:12.1.1-devel-ubuntu20.04 /bin/bash
```

2. Configure the runner by `config.sh`

Run these command inside the container.

```
apt update && apt install -y curl
export RUNNER_ALLOW_RUNASROOT=1
```

Then follow https://github.com/sgl-project/sglang/settings/actions/runners/new?arch=x64&os=linux to run `config.sh`

Notes
- Do not need to specify the runner group
- Give it a name (e.g., `test-sgl-gpu-0`) and some labels (e.g., `unit-test`). The labels can be editted later in Github Settings.
- Do not need to change the work folder.

3. Run the runner by `run.sh`

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