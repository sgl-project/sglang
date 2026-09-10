# SenseNova-U1 Vendored Model Files

The Python files in this directory are adapted from:

- Repository: https://github.com/OpenSenseNova/SenseNova-U1
- Commit: 2f42002f9b819506c9deb44599f0809e30252aba

They were modified for SGLang multimodal_gen native integration.

Importing this package does not load model implementations or register Transformers
Auto classes. The SenseNova-U1 pipeline calls `register()` when loading a model.
Direct users of `AutoConfig` or `AutoModel` must call it explicitly first:

```python
from sglang.multimodal_gen.runtime.models.sensenova_u1 import register

register()
```

Import config classes from `sglang.multimodal_gen.configs.transformers` submodules
and model classes or attention helpers from their respective `modeling_*` modules.
