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

def _random_fill_tensor(t: torch.Tensor, *, low=None, high=None):
    """
    Fill tensor `t` in-place with uniform random values.
    - Floating: U(0,1)
    - Complex: real/imag both U(0,1)
    - Bool: Bernoulli with p=0.5
    - Integer: uniform integer in [low, high)
    """
    device = t.device
    shape = t.shape
    dtype = t.dtype

    # Floating types (float32/64/16 + bfloat16)
    if dtype.is_floating_point:
        gen_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        tmp = torch.rand(shape, device=device, dtype=gen_dtype)
        t.copy_(tmp.to(dtype, copy=False))
        return

    # Complex types
    if dtype.is_complex:
        comp_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.rand(shape, device=device, dtype=comp_dtype)
        imag = torch.rand(shape, device=device, dtype=comp_dtype)
        t.copy_(torch.complex(real, imag).to(dtype))
        return

    # Bool
    if dtype == torch.bool:
        mask = torch.rand(shape, device=device) > 0.5
        t.copy_(mask)
        return

    # Integer types
    if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        info = torch.iinfo(dtype)
        # Default integer range: full dtype range
        if low is None: low = int(info.min)
        if high is None:
            # torch.randint high is exclusive; make maxv+1 if safe
            maxv = int(info.max)
            if maxv - low + 1 <= (1 << 31):
                high = maxv + 1
            else:
                # huge range fallback to a safe 32-bit window
                low = max(low, -2**31)
                high = 2**31 - 1

        if not (low < high):
            raise ValueError(f"invalid integer bounds: low={low}, high={high}")

        rand = torch.randint(low=low, high=high, size=shape, device=device, dtype=torch.int64)
        t.copy_(rand.to(dtype))
        return

    raise TypeError(f"unsupported dtype: {dtype}")

