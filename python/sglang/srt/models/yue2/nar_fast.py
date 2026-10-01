# SPDX-License-Identifier: Apache-2.0
"""Fused NAR acoustic flow matching that reuses the AR session's KV cache.

The AR decode session already holds the exact key/value cache of
``prefix + codec tokens + MUSIC_END`` (the NAR "original chunk" tokens): the
semantic phase wrote it slot by slot. 
"""
from __future__ import annotations

import os
import threading

import torch
import torch.nn.functional as F

from .protocol import MUSIC_END, chunk_ranges

_SGL_KERNEL = None


def _sgl_kernel():
    """Load sgl_kernel once; returns None when unavailable.

    `import sgl_kernel` transitively imports flashinfer, whose cubin version
    check aborts unless FLASHINFER_DISABLE_VERSION_CHECK=1 is set — the most
    common reason this backend appears missing in a fresh environment.
    """
    global _SGL_KERNEL
    if _SGL_KERNEL is None:
        try:
            import sgl_kernel as module

            _SGL_KERNEL = module
        except Exception:
            _SGL_KERNEL = False
    return _SGL_KERNEL or None


def model_fused_nar_weights(model):
    """Persistent fused QKV / gate-up weights for the NAR expert, cached on the model."""
    cached = getattr(model, "_yue2_fused_nar_weights", None)
    if cached is None:
        layers = []
        for layer in model.model.layers:
            qkv = torch.cat(
                [layer.nar_self_attn.q_proj.weight, layer.nar_self_attn.k_proj.weight,
                 layer.nar_self_attn.v_proj.weight], dim=0).contiguous()
            gate_up = torch.cat(
                [layer.nar_mlp.gate_proj.weight, layer.nar_mlp.up_proj.weight], dim=0).contiguous()
            layers.append((qkv, gate_up))
        model._yue2_fused_nar_weights = layers
        cached = layers
    return cached


def _flash_varlen(q, k, v, cu_q, cu_k, max_q, max_k):
    return torch.ops.aten._flash_attention_forward(
        q, k, v, cu_q, cu_k, max_q, max_k, 0.0, False, False)[0]


class SessionNAR:
    """Velocity evaluations for one original acoustic chunk on borrowed AR KV."""

    def __init__(self, model, session, ar_length, noise):
        kernel = _sgl_kernel()
        if kernel is None:
            # NAR stage treats ValueError as "fall back to the reference
            # implementation" — degrade gracefully instead of failing the
            # request when sgl_kernel cannot be imported.
            raise ValueError(
                "sgl_kernel is unavailable; falling back to the reference NAR "
                "(export FLASHINFER_DISABLE_VERSION_CHECK=1 and reinstall "
                "sgl-kernel to enable the fused path)"
            )

        self.model, self.session = model, session

        # TODO (yiakwy) : sgl_kernel seems to be removed from sources, check if it is still needed in the future
        # NOTE (yiakwy) : we verified with sgl_kernel == 0.4.7 and flashinfer (0.7.0.post1)
        self.kernel = kernel

        self.fused = model_fused_nar_weights(model)
        config = model.config
        self.B, self.H = 1, config.hidden_size
        self.HQ, self.HKV, self.HD = (config.num_attention_heads,
                                      config.num_key_value_heads, config.head_dim)
        self.eps = config.rms_norm_eps
        self.device = session.device
        self.dtype = session.dtype
        self.ar_length = int(ar_length)
        self.noise = noise
        self.nar_length = noise.shape[0] + 2
        total = self.ar_length + self.nar_length
        # Per-layer velocity KV: [AR KV | NAR KV], exact length, no masking.
        shape = (1, total, self.HKV, self.HD)
        self.keys = [torch.empty(shape, device=self.device, dtype=self.dtype)
                     for _ in model.model.layers]
        self.values = [torch.empty(shape, device=self.device, dtype=self.dtype)
                       for _ in model.model.layers]
        
        with torch.no_grad():
            for index in range(len(model.model.layers)):
                self.keys[index][0, :self.ar_length].copy_(
                    session.keys[index][0, :self.ar_length])
                self.values[index][0, :self.ar_length].copy_(
                    session.values[index][0, :self.ar_length])
                
        self.cu_q = torch.tensor([0, self.nar_length], dtype=torch.int32, device=self.device)
        self.cu_k = torch.tensor([0, total], dtype=torch.int32, device=self.device)
        self.nar_pos = torch.arange(self.ar_length, total, device=self.device,
                                    dtype=torch.int32)
        
        local = torch.arange(self.nar_length, device=self.device).clamp(
            max=config.max_latent_frames - 1)
        
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.layers = model.model.layers
        self.last = len(self.layers) - 1

    @torch.inference_mode()
    def velocity(self, state, raw_t, shifted=None):
        model = self.model
        kernel = self.kernel
        B, H, HD = self.B, self.H, self.HD
        HQ, HKV, eps = self.HQ, self.HKV, self.eps
        x_nar = F.pad(state, (0, 0, 1, 1))

        if shifted is None:
            shifted = model._shift_t_value(raw_t, self.device, self.dtype)

        residual = model.vae2llm(x_nar[None])
        residual = residual + model.time_embedder(shifted.expand(self.nar_length))[None]
        residual = residual + self.pos_emb

        hidden = kernel.rmsnorm(
            residual.view(B * self.nar_length, H), self.layers[0].nar_input_layernorm.weight, eps
        ).view(B, self.nar_length, H)

        for index, layer in enumerate(self.layers):
            qkv_w, gate_up_w = self.fused[index]
            qkv = F.linear(hidden.view(B * self.nar_length, H), qkv_w)
            kernel.fused_qk_norm_rope(
                qkv, HQ, HKV, HKV, HD, eps,
                layer.nar_self_attn.q_norm.weight, layer.nar_self_attn.k_norm.weight,
                model.config.rope_theta, True, self.nar_pos, 1.0, 0.0, 0.0, 1.0,
            )

            q = qkv[:, : HQ * HD].view(self.nar_length, HQ, HD)
            k = qkv[:, HQ * HD: (HQ + HKV) * HD].view(self.nar_length, HKV, HD)
            v = qkv[:, (HQ + HKV) * HD:].view(self.nar_length, HKV, HD)
            keys, values = self.keys[index], self.values[index]

            keys[0, self.ar_length:] = k
            values[0, self.ar_length:] = v

            h = _flash_varlen(
                q, keys[0], values[0], self.cu_q, self.cu_k,
                self.nar_length, self.ar_length + self.nar_length)
            attn = layer.nar_self_attn.o_proj(h.reshape(self.nar_length, HQ * HD))
            kernel.fused_add_rmsnorm(
                attn, residual.view(B * self.nar_length, H), layer.nar_pre_mlp_layernorm.weight, eps)
            
            # TODO (yiakwy) : fuse linear into rmsnorm
            gate_up = F.linear(attn, gate_up_w)

            mlp = layer.nar_mlp.down_proj(kernel.silu_and_mul(gate_up))
            next_weight = (self.layers[index + 1].nar_input_layernorm.weight
                           if index < self.last else model.model.norm.weight)
            
            kernel.fused_add_rmsnorm(mlp, residual.view(B * self.nar_length, H), next_weight, eps)

            hidden = mlp.view(B, self.nar_length, H)

        return model.llm2vae(hidden)[0, 1:-1]

    # NOTE (yiawy) : main flash flow matching solver
    @torch.inference_mode()
    def solve(self, steps):
        state = self.noise.to(device=self.device, dtype=self.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = state - self.velocity(mid, raw_mid) * dt
        result = state.float().cpu()

        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result


class _VelocityGraph(SessionNAR):
    """SessionNAR whose ``velocity`` is a captured CUDA graph replay.

    The ODE state and the shifted timestep are graph input buffers, so a
    replay never bakes request-specific scalars into the capture.
    """

    def __init__(self, model, session, ar_length, noise):
        super().__init__(model, session, ar_length, noise)
        self.state_buf = self.noise.to(device=self.device, dtype=self.dtype).clone()
        self.shifted_buf = torch.zeros(1, device=self.device, dtype=self.dtype)

        # TODO (yiakwy) : remove warmup
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup):
            for _ in range(2):
                self._velocity_impl(self.state_buf)
        torch.cuda.current_stream(self.device).wait_stream(warmup)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output_buf = self._velocity_impl(self.state_buf)

    def load(self, session, ar_length, noise):
        """Refresh the borrowed AR KV and the noise buffer for a new request."""
        self.ar_length = int(ar_length)
        self.noise = noise
        self.nar_length = noise.shape[0] + 2
        with torch.no_grad():
            for index in range(len(self.layers)):
                self.keys[index][0, :self.ar_length].copy_(
                    session.keys[index][0, :self.ar_length])
                self.values[index][0, :self.ar_length].copy_(
                    session.values[index][0, :self.ar_length])
        self.state_buf.copy_(noise.to(device=self.device, dtype=self.dtype))

    @torch.inference_mode()
    def velocity(self, state, raw_t, shifted=None):
        self.state_buf.copy_(state)
        self.shifted_buf.copy_(self.model._shift_t_value(raw_t, self.device, self.dtype))
        self.graph.replay()
        return self.output_buf

    @torch.inference_mode()
    def _velocity_impl(self, state):
        return super().velocity(state, None, self.shifted_buf)


class VelocityGraphPool:
    """Pools captured velocity graphs keyed by (ar_length, latent_frames)."""

    def __init__(self, max_shapes: int = 8):
        self.max_shapes = int(max_shapes)
        self._graphs: dict[tuple[int, int], _VelocityGraph] = {}
        self._lock = threading.Lock()

    def acquire(self, model, session, ar_length, noise):
        key = (int(ar_length), int(noise.shape[0]))
        with self._lock:
            graph = self._graphs.get(key)
            if graph is None and len(self._graphs) < self.max_shapes:
                graph = _VelocityGraph(model, session, ar_length, noise)
                self._graphs[key] = graph
                return graph
        if graph is not None:
            graph.load(session, ar_length, noise)
            return graph
        return SessionNAR(model, session, ar_length, noise)

    def stats(self) -> dict:
        with self._lock:
            return {"shapes": sorted(self._graphs.keys())}


default_velocity_pool = VelocityGraphPool()


@torch.inference_mode()
def synthesize_from_session(model, session, prefix_len, codec, seed, fed_codec, steps=32,
                            context=None, cancelled=None):
    """Fast-path NAR for single-chunk songs whose AR KV already lives in ``session``.

    Feeds the unfed codec tail plus ``MUSIC_END`` through the session, then
    solves the flow matching on a borrowed copy of the AR KV. Returns CPU
    FP32 ``[frames, 64]`` latents.
    """
    from .protocol import CODEC_OFFSET, CONTEXT

    context = context or CONTEXT
    ranges = chunk_ranges(len(codec), prefix_len, context)
    if len(ranges) != 1:
        raise ValueError("fast NAR path supports single-chunk songs only")
    start, end = ranges[0]
    if start != 0 or end != len(codec):
        raise ValueError("fast NAR path requires full codec coverage")
 
    
    for value in [value + CODEC_OFFSET for value in codec[fed_codec:]] + [MUSIC_END]:
        session.step(torch.tensor([[value]], dtype=torch.long, device=session.device))

    # TODO (yiakwy) : move to setup
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)

    if os.environ.get("SGLANG_YUE2_VELOCITY_GRAPH", "1") == "1":
        engine = default_velocity_pool.acquire(model, session, prefix_len + len(codec) + 1, noise)
    else:
        engine = SessionNAR(model, session, prefix_len + len(codec) + 1, noise)
    try:
        return engine.solve(steps)
    finally:
        if isinstance(engine, SessionNAR) and not isinstance(engine, _VelocityGraph):
            engine.keys = engine.values = None
