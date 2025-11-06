# Attention Kernel Used in sgl-diffusion

## VMoBA: Mixture-of-Block Attention for Video Diffusion Models (VMoBA)

### Installation
Please ensure that you have installed FlashAttention version **2.7.1 or higher**, as some interfaces have changed in recent releases.

### Usage

You can use `moba_attn_varlen` in the following ways:

**Install from source:**
```bash
python setup.py install
```

**Import after installation:**
```python
from vmoba import moba_attn_varlen
```

**Or import directly from the project root:**
```python
from csrc.attn.vmoba_attn.vmoba import moba_attn_varlen
```

### Verify if you have successfully installed

```bash
python csrc/attn/vmoba_attn/vmoba/vmoba.py
```
