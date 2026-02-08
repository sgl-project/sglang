# Blackwell Distributed GEMM

## Requirements

### Build
Make sure to set up CUTLASS with
support for [Programmatic Dependent Launch (PDL)](../../media/docs/dependent_kernel_launch.md),
that is with the `CUTLASS_ENABLE_GDC_FOR_SM100` flag.

```bash
cmake $PATH -DCUTLASS_NVCC_ARCHS="100a" -DCUTLASS_ENABLE_GDC_FOR_SM100=1
```

### Minimum software

Like all other CUTLASS examples, the NVIDIA driver, runtime, and CUDA Toolkit are required.
This example specifically requires CUDA Toolkit 12.8 or newer, since that is the first version
supporting the Blackwell architecture.

### Hardware / driver settings

This example requires Blackwell GPUs with NVLink network.

If you're not sure, first run the following command and make sure your GPU
compute capability is 10.0:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Sample output:

```
name, compute_cap
NVIDIA B200, 10.0
NVIDIA B200, 10.0
NVIDIA B200, 10.0
NVIDIA B200, 10.0
NVIDIA B200, 10.0
NVIDIA B200, 10.0
NVIDIA B200, 10.0
NVIDIA B200, 10.0
```


Then you should make sure there is an NVLink network by checking the GPU network topology,
and making sure there's `NV*` links between every pair of GPUs:

```bash
nvidia-smi topo -m
```

Sample output:

```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18
GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18
GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18
GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18
GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18
GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18
GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X
```

Finally, check if the driver enables peer to peer access, which should usually be the case,
but it's good to check anyway:

```bash
nvidia-smi topo -p2p r
```

Sample output:

```
       GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0   X       OK      OK      OK      OK      OK      OK      OK
GPU1   OK      X       OK      OK      OK      OK      OK      OK
GPU2   OK      OK      X       OK      OK      OK      OK      OK
GPU3   OK      OK      OK      X       OK      OK      OK      OK
GPU4   OK      OK      OK      OK      X       OK      OK      OK
GPU5   OK      OK      OK      OK      OK      X       OK      OK
GPU6   OK      OK      OK      OK      OK      OK      X       OK
GPU7   OK      OK      OK      OK      OK      OK      OK      X
```
