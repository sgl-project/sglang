import pytest
import sgl_kernel
import torch

device = "cuda" if torch.cuda.is_available() else None


# Data Generation
def datagen(
    dtype,
    param_type,
    batch,
    seq,
    frame,
    hidden_dim,
    use_affine,
    use_bias,
    eps,
    norm_type,
    gate_shape,
    scale_shift_shape,
):
    residual = torch.randn(batch, seq, hidden_dim, dtype=dtype, device=device)
    x = torch.randn(batch, seq, hidden_dim, dtype=dtype, device=device)
    if gate_shape == "1":
        gate = None
    elif gate_shape == "1D":
        gate = torch.randn(1, hidden_dim, dtype=dtype, device=device)
    elif gate_shape == "BF1D":
        if seq % frame != 0:
            pytest.skip(f"seq ({seq}) must be divisible by frame ({frame}).")
        gate = torch.randn(batch, frame, 1, hidden_dim, dtype=dtype, device=device)
    elif gate_shape == "B1D":
        gate = torch.randn(batch, 1, hidden_dim, dtype=dtype, device=device)
    else:
        raise ValueError("Unknown gate shape.")
    norm_weight, norm_bias = None, None
    if use_affine:
        norm_weight = torch.randn(hidden_dim, dtype=param_type, device=device)
        if norm_type == "layer" and use_bias:
            norm_bias = torch.randn(hidden_dim, dtype=param_type, device=device)
    if "1" == scale_shift_shape:
        scale = torch.tensor(1.0, dtype=dtype, device=device)
        shift = torch.tensor(1.0, dtype=dtype, device=device)
    elif scale_shift_shape == "BD":
        scale = torch.randn(batch, hidden_dim, dtype=dtype, device=device)
        shift = torch.randn(batch, hidden_dim, dtype=dtype, device=device)
    elif scale_shift_shape == "1D":
        scale = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        shift = torch.randn(1, hidden_dim, dtype=dtype, device=device)
    elif scale_shift_shape == "BSD":
        scale = torch.randn(batch, seq, hidden_dim, dtype=dtype, device=device)
        shift = torch.randn(batch, seq, hidden_dim, dtype=dtype, device=device)
    elif scale_shift_shape == "B1D":
        scale = torch.randn(batch, 1, hidden_dim, dtype=dtype, device=device)
        shift = torch.randn(batch, 1, hidden_dim, dtype=dtype, device=device)
    elif scale_shift_shape == "1SD":
        scale = torch.randn(1, seq, hidden_dim, dtype=dtype, device=device)
        shift = torch.randn(1, seq, hidden_dim, dtype=dtype, device=device)
    elif scale_shift_shape == "11D":
        scale = torch.randn(1, 1, hidden_dim, dtype=dtype, device=device)
        shift = torch.randn(1, 1, hidden_dim, dtype=dtype, device=device)
    elif scale_shift_shape == "BF1D":
        if seq % frame != 0:
            pytest.skip(f"seq ({seq}) must be divisible by frame ({frame}).")
        scale = torch.randn(batch, frame, 1, hidden_dim, dtype=dtype, device=device)
        shift = torch.randn(batch, frame, 1, hidden_dim, dtype=dtype, device=device)
    return (
        residual,
        x,
        gate,
        norm_weight,
        norm_bias,
        shift,
        scale,
        eps,
        norm_type == "rms",
    )


# Reference
def scale_residual_norm_scale_shift_ref(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    norm_type: bool,
):
    # 1. residual add
    if isinstance(gate, torch.Tensor):
        if gate.dim() == 4:
            # gate.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = gate.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            residual_out = residual + (
                x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
            ).flatten(1, 2)
        else:
            # gate.shape: [batch_size, 1, inner_dim]
            residual_out = residual + x * gate
    else:
        gate = 1
        residual_out = residual + x * gate
    # 2. normalize
    if norm_type == False:  # LayerNorm
        mean = residual_out.mean(dim=-1, keepdim=True)
        var = residual_out.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (residual_out - mean) / torch.sqrt(var + eps)
    elif norm_type == True:  # RMSNorm
        rms = residual_out.pow(2).mean(dim=-1, keepdim=True)
        normalized = residual_out / torch.sqrt(rms + eps)
    # 3. apply affine transform if given
    if norm_weight is not None and norm_bias is not None:
        normalized = normalized * norm_weight + norm_bias
    elif norm_weight is not None:
        normalized = normalized * norm_weight
    # 4. apply scale/shift if given
    batch, seq_len, hidden_dim = x.shape
    if scale.ndim <= 3:
        if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
            # (), (1) → (B, S, D)
            scale = scale.expand(batch, seq_len, hidden_dim)
            shift = shift.expand(batch, seq_len, hidden_dim)
        elif scale.ndim == 2 and scale.shape in [
            (1, hidden_dim),
            (batch, hidden_dim),
        ]:
            # (B, D) or (1, D) → (B, S, 1, D)
            scale = scale[:, None, :].expand(batch, seq_len, hidden_dim)
            shift = shift[:, None, :].expand(batch, seq_len, hidden_dim)
        elif scale.ndim == 3 and scale.shape in [
            (batch, seq_len, hidden_dim),
            (batch, 1, hidden_dim),
            (1, seq_len, hidden_dim),
            (1, 1, hidden_dim),
        ]:
            # (B, S, D), (B, 1, D), (1, S, D), (1, 1, D) → (B, S, 1, D)
            scale = scale.expand(batch, seq_len, hidden_dim)
            shift = shift.expand(batch, seq_len, hidden_dim)
        normalized = normalized * (1.0 + scale) + shift
    elif scale.ndim == 4 and scale.shape == (batch, scale.shape[1], 1, hidden_dim):
        num_frames = scale.shape[1]
        frame_seqlen = normalized.shape[1] // num_frames
        normalized = (
            normalized.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1.0 + scale)
            + shift
        ).flatten(1, 2)
    return normalized, residual_out


compiled_scale_residual_norm = torch.compile(scale_residual_norm_scale_shift_ref)


def _run_test(
    *,
    dtype=torch.float32,
    batch=1,
    seq=2048,
    frame=4,
    hidden_dim=1536,
    use_affine=True,
    use_bias=True,
    eps=1e-6,
    norm_type="layer",
    gate_shape="1D",
    scale_shift_shape="B1D",
):
    if device is None:
        pytest.skip("No cuda device available for this test")

    param_type = dtype
    input_data = datagen(
        dtype,
        param_type,
        batch,
        seq,
        frame,
        hidden_dim,
        use_affine,
        use_bias,
        eps,
        norm_type,
        gate_shape,
        scale_shift_shape,
    )
    mod_ref, resi_out_ref = compiled_scale_residual_norm(*input_data)
    mod, resi_out = sgl_kernel.scale_residual_norm_scale_shift(*input_data)

    if dtype == torch.float32:
        torch.testing.assert_close(mod, mod_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(resi_out, resi_out_ref, rtol=1e-5, atol=1e-5)
    elif dtype == torch.float16:
        torch.testing.assert_close(mod, mod_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(resi_out, resi_out_ref, rtol=1e-2, atol=1e-2)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(mod, mod_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(resi_out, resi_out_ref, rtol=1e-2, atol=1e-2)
    else:
        raise ValueError(f"Not implement data type: {dtype}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_type", ["layer", "rms"])
def test_scale_residual_norm_scale_shift_dtype(dtype, norm_type):
    _run_test(dtype=dtype, norm_type=norm_type)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_scale_residual_norm_scale_shift_batch(batch):
    _run_test(batch=batch)


@pytest.mark.parametrize("seq", [83, 1024, 2047, 32760])
def test_scale_residual_norm_scale_shift_seq(seq):
    _run_test(seq=seq)


@pytest.mark.parametrize("frame", [4, 8])
def test_scale_residual_norm_scale_shift_frame(frame):
    _run_test(frame=frame)


@pytest.mark.parametrize("hidden_dim", [83, 1024, 1536, 3072, 4096])
def test_scale_residual_norm_scale_shift_hidden_dim(hidden_dim):
    _run_test(hidden_dim=hidden_dim)


@pytest.mark.parametrize("use_affine", [False, True])
def test_scale_residual_norm_scale_shift_affine(use_affine):
    _run_test(use_affine=use_affine)


@pytest.mark.parametrize("use_bias", [True])
def test_scale_residual_norm_scale_shift_bias(use_bias):
    _run_test(use_bias=use_bias)


@pytest.mark.parametrize("norm_type", ["layer", "rms"])
def test_scale_residual_norm_scale_shift_norm_type(norm_type):
    _run_test(norm_type=norm_type)


@pytest.mark.parametrize("gate_shape", ["1", "1D", "B1D", "BF1D"])
def test_scale_residual_norm_scale_shift_gate_shape(gate_shape):
    _run_test(gate_shape=gate_shape)


@pytest.mark.parametrize(
    "scale_shift_shape", ["1", "1D", "BD", "BSD", "B1D", "1SD", "11D", "BF1D"]
)
def test_scale_residual_norm_scale_shift_scale_shift_shape(scale_shift_shape):
    _run_test(scale_shift_shape=scale_shift_shape)


if __name__ == "__main__":
    pytest.main([__file__])
