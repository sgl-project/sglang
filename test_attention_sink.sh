#!/bin/bash

# To set up the sglang server:
# python -m sglang.launch_server --model Qwen/Qwen3-8B --attention-backend fa3 --disable-cuda-graph

curl http://localhost:30000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "messages": [
        {"role": "user", "content": "There are $n$ values of $x$ in the interval $0<x<2\\pi$ where $f(x)=\\sin(7\\pi\\cdot\\sin(5x))=0$. For $t$ of these $n$ values of $x$, the graph of $y=f(x)$ is tangent to the $x$-axis. Find $n+t$. Put your answer in \\boxed{}. For multiple-choice questions, only put the choice letter in the box."}
    ],
    "max_tokens": 16384
}'
