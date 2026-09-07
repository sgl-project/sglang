# DeepSeek-V4.1 encoding golden fixtures

WARNING: these `test_input_N.json` / `test_output_N.txt` pairs are vendored
verbatim from the DeepSeek-V4.1 reference encoder drop. They pin the sglang
encoder byte-for-byte against that reference while it is being brought up.
They MUST be replaced with self-authored cases before any upstream PR.

Case semantics (mirrored by `test_encoding_dsv41.py`):

- A case is either a bare message list or an object with `messages` plus
  optional `tools`, `thinking_mode`, `reasoning_effort`, `context`.
- Case-level `tools` are attached to `messages[0]`.
- A bare message list encodes in `chat` mode.
- The `.txt` goldens have no trailing newline; `end-of-file-fixer` is excluded
  for this directory in `.pre-commit-config.yaml`.
