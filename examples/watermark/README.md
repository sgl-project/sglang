# Watermark detection server

The reference HTTP server wraps the in-tree detector; keys and the context window come only from the server's `--watermark-config` JSON (a file path or inline object), never request bodies.
Run: `python examples/watermark/detection_server.py --watermark-config /run/secrets/sglang-watermark.json --tokenizer Qwen/Qwen3-8B`.
Token IDs: `curl localhost:8000/detect -H 'Content-Type: application/json' -d '{"token_ids":[1,2,3,4,5]}'`.
Text requests use `{"text":"Text to inspect"}` and may set a `tokenizer` identifier when no default was provided; protect this reference endpoint because tokenizer loading and detector scores are externally observable.
