Synthetic Kimi vocabulary: the shared fixture generates 256 byte tokens, then
adds `merges.txt` in rank order and the configured protocol markers. Every
multi-byte token must be reachable by merging two lower-ranked tokens (the
intermediate merges are listed too), as fastokens refuses a vocabulary with
unreachable tokens; special-token ids in `tokenizer_config.json` start past
the last merge.
`prompts.json` records token counts and SHA-256 of little-endian u32 token IDs
from SGLang's `_encode_messages` and `moonshotai/Kimi-K3` revision
`f831ab66814297da540d832a5235f8e904f29d06`. Regenerate in a SGLang Python environment:

```sh
python tests/scripts/generate_kimi_parity.py
```
