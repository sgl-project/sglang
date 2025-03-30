# Motivation

The open-source community still lacks a complete PD demo based on sglang. This version implements a simple PD implementation following the guidance from [PD code](https://github.com/sgl-project/sglang/pull/4654).

Based on Python pyverbs library

## Limitations

* Currently a draft submission, demonstrating interface completeness
* Based on Python rdma-core verbs library
* Bootstrap Server uses HttpServer as an example, only to prove the workflow
* Current Memory operations (Memory Region operations) may have inappropriate aspects that could affect performance, requiring community review
* Code redundancy is present, please be gentle with criticism

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