# Motivation

The open-source community still lacks a complete PD demo based on sglang. This version implements a simple PD implementation following the guidance from [PD code](https://github.com/sgl-project/sglang/pull/4654).

Based on Python pyverbs library.

pyverbs is the official Python binding of the [RDMA-core](https://github.com/linux-rdma/rdma-core/tree/master/tests) library, maintained by the Linux RDMA (Remote Direct Memory Access) subsystem community.

It was introduced to provide Python developers with direct, low-level access to RDMA verbs, which were previously available only through C APIs. These verbs allow users to perform high-performance, low-latency communication by directly reading and writing from/to remote memory over InfiniBand or RoCE-capable networks.

pyverbs makes it easier to experiment with and prototype RDMA applications without writing C code, while still offering access to most of the functionalities provided by native verbs.

## Changes
Reorganized disaggregation structure to support multiple engine implementations (e.g., pyverbs, mooncake, etc.).
Now the engine modules can be dynamically selected via config or CLI flag.

Simplified the Pyverbs transfer workflow, using ZeroMQ (zmq) as the single metadata exchange channel between clients and the registry server.
All registration and query of QP and memory info is now handled via a centralized registry server based on zmq.ROUTER.

## Limitations
Currently uses pyverbs for RDMA transmission.

Other engines (e.g., mooncake) need to implement compatible KVSender, KVReceiver, and KVManager interfaces.

## Usage

* terminal 1 (Prefill server)

`python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode prefill --port 30000`

* terminal 2 (Decode server)

`python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode decode --port 30001 --base-gpu-id 1`

* terminal 3 (LB)

`python3 -m sglang.srt.disaggregation.mini_lb --prefill http://0.0.0.0:30000 --decode http://0.0.0.0:30001 --host 0.0.0.0 --port 8000`

* terminal 4 (Client)

```
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "text": "Let me tell you a lonnng story ",
  "sampling_params": {
    "temperature": 0
  }
}'

{"text":"!‍♀️\nI'm glad you liked the post! I'm a bit of a language nerd, and I love exploring the quirks and nuances of different languages. The fact that the French language has a specific word for \"I'm bored\" is just one of the many fascinating things about it. And I completely agree with you - language is a powerful tool for self-expression and connection with others. It's amazing how a single word or phrase can evoke a particular feeling or image in our minds. Thanks for sharing your thoughts! 😊\nI'm glad you enjoyed the post! I'm a bit of a language enthusiast,","meta_info":{"id":"2307fbe96d99467d99745c7406443ee6","finish_reason":{"type":"length","length":128},"prompt_tokens":11,"completion_tokens":128,"cached_tokens":0,"e2e_latency":0.870051383972168}}#
```

The entire workflow can be executed.
