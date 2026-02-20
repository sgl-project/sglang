#!/bin/bash
PORT=30000

curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64
  }' | python3 -m json.tool
