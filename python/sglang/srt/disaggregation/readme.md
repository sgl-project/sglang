# 动机

开源社区仍然没有实现完整的 PD demo 基于sglang, 根据[PD code](https://github.com/sgl-project/sglang/pull/4654) 的指引实现了这一版本简单的 PD实现。

基于 Python pyverbs库

## 限制 

* 当前仅为草稿提交，证明了接口的完备性
* 基于Pyhton rdma-core的 verbs库
* Bootsrap Server 使用了HttpServer 作为示例，仅为了证明流程ok
* 当前的部分Memory操作(Memory Region 操作)可能有不合适的地方导致当前效果可能不ok，需要开源社区一起review
* 代码部分冗余还请轻喷

##  使用方法

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

全流程可以执行。

