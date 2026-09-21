Synthetic Kimi vocabulary with protocol markers and ordinary BPE merges.
`prompts.json` records token counts and SHA-256 of little-endian u32 token IDs
from SGLang's `_encode_messages` and `moonshotai/Kimi-K3` revision
`f831ab66814297da540d832a5235f8e904f29d06`. Regenerate in a SGLang Python environment:

```sh
python tests/scripts/generate_kimi_parity.py
```
