---
paths:
  - "**/*.py"
---

# Don't use `getattr` / `hasattr` for defensive access

Over-defensive `getattr(obj, "field", default)` / `hasattr(obj, "field")` hide
errors and defeat strict type checking. If a field is always present, accessing
it defensively is confusing and masks real bugs. Prefer:

1. **`isinstance` for type narrowing** — check the type, then access fields directly:

   ```python
   if (
       isinstance(obj, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput))
       and obj.mm_inputs
   ):
   ```
   (see `python/sglang/srt/managers/mm_utils.py`)

2. **Always set the field (to `None` if needed), then do a `None` check** — the
   field should always exist, so a `None` / non-`None` check is enough:

   ```python
   obj.field = None   # in __init__ / construction
   ...
   if obj.field is not None:
       ...
   ```

Bad — `server_args` always has `revision`, so `getattr` is misleading and swallows
a real `AttributeError` if the field is ever renamed:

```python
revision=getattr(server_args, "revision", None),   # BAD
revision=server_args.revision,                     # GOOD
```
(see `python/sglang/srt/managers/template_detection.py`)
