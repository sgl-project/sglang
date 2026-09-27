# Watermark detection server

The reference HTTP server wraps the in-tree detector; keys come only from `SGLANG_WATERMARK_KEY` and optional `SGLANG_WATERMARK_KEY_B`, never request bodies.
Run: `SGLANG_WATERMARK_KEY=0123456789abcdef python examples/watermark/detection_server.py --tokenizer Qwen/Qwen3-8B`.
Token IDs: `curl localhost:8000/detect -H 'Content-Type: application/json' -d '{"token_ids":[1,2,3,4,5]}'`.
Text requests use `{"text":"Text to inspect"}` and may set a `tokenizer` identifier when no default was provided; protect this reference endpoint because tokenizer loading and detector scores are externally observable.
