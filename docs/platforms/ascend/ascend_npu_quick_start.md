# Ascend NPU Quickstart

## Prerequisites

### Supported Devices

- Atlas 800I A2 inference series (Atlas 800I A2)
- Atlas 800I A3 inference series (Atlas 800I A3)

## Setup environment using container

__Notice:__ The following commands are based on Atlas 800I A3 machines. If you are using Atlas 800I A2, some changes are needed.

- The image tag needs to be `main-cann8.5.0-a3` for Atlas 800I A3 and `main-cann8.5.0-910b` for Atlas 800I A2.
- The device mapping in `docker run` command needs to be changed to `davinci[0-7]` for Atlas 800I A2.

```shell
# For Atlas 800I A3
export IMAGE=quay.io/ascend/sglang:main-cann8.5.0-a3

docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin \
    --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule \
    --volume ~/.cache/:/root/.cache/ \
    --entrypoint=bash \
    $IMAGE
```

## Usage

The SGLang server is installed in the container by default. You can use `pip show sglang` to check the version.

### Start SGLang server

SGLang will automatically download the model from Hugging Face.

```shell
# Set HF_ENDPOINT to a mirror site if network is not available
export HF_ENDPOINT=https://hf-mirror.com

# Set your own HF_TOKEN to download restricted models
export HF_TOKEN=<secret>

# Start SGLang server
# It may take several minutes to download the model on the first run
sglang serve --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend ascend &
```

If you see output like the following, the server is running.

```log
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
The server is fired up and ready to roll!
```

### Send a test request

You can do inference using the server:

```shell
curl -X POST http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16
        }
    }'
```

If the "text" field in the response contains "Paris", the server is working as expected.

### Stop server and exit container

The SGLang server is running as a background process. You can send a `SIGINT` signal to stop it.

```shell
SGLANG_PID=$(pgrep -f "sglang serve")
kill -SIGINT $SGLANG_PID
```

The output should be like the following:

```log
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [25310]
```

The server has now stopped. You can verify it with `ps -ef | grep sglang`, then exit the container by pressing `Ctrl+D`.
