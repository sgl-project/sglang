class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner

    def handle(self, action: str):
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_param":
            self._reset_param()
        elif action == "compare":
            self._compare()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        TODO

    def _reset_param(self):
        for name, param in self._model_state():
            TODO

    def _compare(self):
        TODO

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model_runner.model.named_parameters()

import torch

def fill_tensor_with_random(t: torch.Tensor, *, low=None, high=None, dist='uniform'):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        gen_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        if dist == 'normal':
            tmp = torch.randn(shape, device=device, dtype=gen_dtype)
        else:
            tmp = torch.rand(shape, device=device, dtype=gen_dtype)
        t.copy_(tmp.to(dtype, copy=False))
        return

    # Complex types
    if dtype.is_complex:
        # choose real dtype for components
        comp_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        if dist == 'normal':
            real = torch.randn(shape, device=device, dtype=comp_dtype)
            imag = torch.randn(shape, device=device, dtype=comp_dtype)
        else:
            real = torch.rand(shape, device=device, dtype=comp_dtype)
            imag = torch.rand(shape, device=device, dtype=comp_dtype)
        comp = torch.complex(real, imag).to(dtype)
        t.copy_(comp)
        return

    # Bool
    if dtype == torch.bool:
        # Bernoulli p=0.5
        mask = torch.rand(shape, device=device) > 0.5
        t.copy_(mask)
        return

    # Integer types (signed/unsigned)
    # Use torch.iinfo to get dtype range; pick sensible subrange if full-range is too large for randint.
    if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        info = torch.iinfo(dtype)
        minv = int(info.min)
        maxv = int(info.max)

        if low is None:
            low = minv
        if high is None:
            # torch.randint's high is exclusive; default to maxv+1 if safe
            # but if range is gigantic (e.g., full uint64-like), clamp to a safe 32-bit window
            try:
                range_size = maxv - minv + 1
            except OverflowError:
                range_size = 1 << 63  # fallback big
            if range_size <= (1 << 31):
                high = maxv + 1
            else:
                # choose a centered 32-bit window to avoid overflowing torch.randint's internal limits
                low = max(minv, -2**31)
                high = min(maxv, 2**31 - 1) + 1

        # torch.randint requires low < high
        if not (low < high):
            raise ValueError(f"invalid integer bounds: low={low}, high={high}")

        # produce as int64 then cast if necessary (torch.randint supports dtype arg for integer types,
        # but generating in int64 then cast is robust)
        rand = torch.randint(low=low, high=high, size=shape, device=device, dtype=torch.int64)
        # cast to target dtype and copy
        t.copy_(rand.to(dtype))
        return

    raise TypeError(f"unsupported dtype: {dtype}")
