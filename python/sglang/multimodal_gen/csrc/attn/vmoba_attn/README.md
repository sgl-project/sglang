# Attention Kernel Used in SGLang diffusion

## VMoBA: Mixture-of-Block Attention for Video Diffusion Models (VMoBA)

### Installation
Please ensure that you have installed FlashAttention version **2.7.1 or higher**, as some interfaces have changed in recent releases.

### Intel XPU

On Intel XPU, VMoBA uses `sgl-kernel-xpu` rather than the CUDA FlashAttention
package. This path is inference-only and requires FP16 or BF16, plus a kernel
build that honours K/V row strides and returns correct multi-query
`softmax_lse`. `sgl_kernel` enforces Battlemage (BMG) or later on every call;
the two contracts are covered by
`test/unit/test_vmoba_xpu_integration.py::TestVMoBAXpuKernelContract`.

Select `--attention-backend vmoba_attn` and provide the six chunking parameters
under `moba_config` in `--attention-backend-config`. For example, for a video
patch grid of `(21, 30, 52)`:

```text
--attention-backend vmoba_attn \
--attention-backend-config '{"moba_config":{"temporal_chunk_size":1,"temporal_topk":3,"spatial_chunk_size":[2,4],"spatial_topk":3,"st_chunk_size":[3,2,4],"st_topk":3,"moba_select_mode":"topk","moba_threshold_type":"query_head"}}'
```

These are illustrative settings, not quality- or performance-tuned defaults.
Spatial and spatiotemporal chunk dimensions must divide the corresponding
video patch dimensions. The denoising stage builds VMoBA metadata from this
configuration, the current step, and the model's latent shape and patch size.
Explicit VMoBA selection rejects unsupported dtypes or missing XPU attention
dependencies instead of silently selecting dense attention. Peak memory is
well above dense attention's, so size the card for the chosen resolution.

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
