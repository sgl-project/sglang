# Dynamo Crate SHA Pin

The three dynamo crates (`dynamo-protocols`, `dynamo-tokenizers`, `dynamo-parsers`) used by sgl-router are pinned to a single SHA on `ai-dynamo/dynamo`.

**Current pin:** `1efdd4dcb901caeae636131321094090d252c8d6`
**Pinned at:** 2026-05-14
**Pinned by:** Kangyan
**Reason:** Initial M0 scaffold. Picked as the most recent green CI commit on `origin/main` that compiles `cargo check -p dynamo-{protocols,tokenizers,parsers}` cleanly locally.

## Bumping

1. Pick a newer SHA from a green `ai-dynamo/dynamo` `main` CI run (`gh run list --repo ai-dynamo/dynamo --branch main --limit 20 --json conclusion,headSha`).
2. `cd ~/dynamo && git checkout <new-sha> && cargo check -p dynamo-protocols -p dynamo-tokenizers -p dynamo-parsers` — must pass.
3. Update `sgl-router/Cargo.toml` `rev = "..."` for all three deps + this file.
4. `cd sgl-router && cargo update && cargo build && cargo test` — must pass.
5. Open PR titled `chore(sgl-router): bump dynamo pin to <short-sha>`.

## Baseline CI measurements

_To be populated by Task 9 of M0 (open scaffold PR + observe CI). Per spec §11 Decision #5, the ceiling is 15 min cold-cache for tier-2 (build+test); above that we escalate to Fork B2 (pre-built deps base image) in this same milestone._

| Metric | Value | Date |
|---|---|---|
| Tier-2 cold-cache build duration | (pending) | |
| Tier-2 warm-cache build duration | (pending) | |
