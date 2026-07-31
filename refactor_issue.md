1. IMAGE/AUDIO 走 scheduler → batch_encode 无 global cache。
2. encode() / batch_encode() 双路径：cache、video aux、health 能力不对齐；能否 batch 的逻辑放在 scheduler（如 video batch-of-1），不该 API 分叉。
3. Tokenizer 侧 /encode 基本 fire-and-forget，_check_encoder_responses 失败不传导，与 scheduler 完成路径脱节可以简化
4. /encode handler 过重（DP/本地/三套 transfer）；/send 可以明确为 mooncake 专用。

Problems:

1. IMAGE/AUDIO via scheduler → batch_encode skip global cache, so batch silently disables cache.
2. Dual encode() / batch_encode() APIs misalign cache, video aux, and health; fuse-or-not should be scheduler policy (e.g. video batch-of-1), not an API fork.
3. Tokenizer /encode is largely fire-and-forget; _check_encoder_responses failures are not propagated and are decoupled from scheduler completion.
4. /encode handler is overloaded (DP/local/three transfer backends); /send should be explicitly mooncake-only.
