# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch AR decode graphs for one request or its two CFG branches.

The graph predicts branch logits only. The caller combines logits and samples
once, then passes that same token to ``step``. Prefixes retain independent RoPE
positions and cache slots. No vLLM, Triton, or custom extension is imported.

``GraphSessionPool`` keeps captured graphs alive across requests: capturing
the 28-layer decode graph costs ~35 ms, which every request would otherwise
pay twice (ABC + semantic phases).
"""
from __future__ import annotations
import os
import threading
from numbers import Integral

import torch
import torch.nn.functional as F


def _fused_enabled() -> bool:
    return os.environ.get("SGLANG_YUE2_FUSED_DECODE", "1") == "1"


def model_fused_weights(model):
    """Persistent per-layer fused QKV / gate-up weights (AR expert), cached on the model.

    The copies are only used by the fused decode path; the original modules
    keep serving prefill and the NAR branch.
    """
    cached = getattr(model, "_yue2_fused_weights", None)
    if cached is None:
        layers = []
        for layer in model.model.layers:
            qkv = torch.cat(
                [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
                 layer.self_attn.v_proj.weight], dim=0).contiguous()
            gate_up = torch.cat(
                [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0).contiguous()
            layers.append((qkv, gate_up))
        model._yue2_fused_weights = layers
        cached = layers
    return cached


class _PrefixCache:
    """A single branch view used only by the original eager HF prefill."""
    def __init__(self, keys, values, branch):
        self.key_cache = [value[branch:branch+1].transpose(1, 2) for value in keys]
        self.value_cache = [value[branch:branch+1].transpose(1, 2) for value in values]
        self.seen = 0

    def get_seq_length(self, layer_idx=0):
        return self.seen

    def update(self, key, value, layer_idx, cache_kwargs=None):
        end = self.seen + key.shape[2]
        if end > self.key_cache[layer_idx].shape[2]:
            raise ValueError("Prefix exceeds preallocated KV capacity")
        self.key_cache[layer_idx][:, :, self.seen:end].copy_(key)
        self.value_cache[layer_idx][:, :, self.seen:end].copy_(value)
        if layer_idx == len(self.key_cache) - 1:
            self.seen = end
        return self.key_cache[layer_idx][:, :, :end], self.value_cache[layer_idx][:, :, :end]


class GraphAR:
    """Fixed-capacity decode for batch one or a synchronized CFG branch pair.

    ``prefill()`` predicts the first token. At most ``max_tokens-1`` calls to
    ``step(token)`` predict the remaining tokens. Returned graph logits share
    output storage and remain valid until the next step. ``capture=False`` is
    an eager diagnostic path for CPU/tiny-model correctness checks.
    """
    def __init__(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto",
                 fuse_projections=False, sampler=None):
        if model.training:
            raise ValueError("GraphAR requires model.eval()")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        prefixes = [list(prefix) for prefix in prefixes]
        if len(prefixes) not in {1, 2}:
            raise ValueError("GraphAR supports one request or exactly two CFG branches")
        config = model.config
        for prefix in prefixes:
            if not prefix or any(isinstance(token, bool) or not isinstance(token, Integral) or
                                 not 0 <= token < config.vocab_size for token in prefix):
                raise ValueError("Prefixes require valid integer token IDs")
            if len(prefix) + max_tokens > config.max_position_embeddings:
                raise ValueError("Prefix plus generation budget exceeds model context; no length was shortened")
        weight = model.model.embed_tokens.weight
        self.model, self.device, self.dtype = model, weight.device, weight.dtype
        linears = [module for layer in model.model.layers for module in
                   (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                    layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)]
        if getattr(model, "_yue2_fp8_originals", None) or any(
                not isinstance(module, torch.nn.Linear) or module.weight.dtype != self.dtype for module in linears):
            raise ValueError("GraphAR only supports unquantized standard AR Linear layers; use eager for FP8")
        if capture and self.device.type != "cuda":
            raise ValueError("CUDA graphs require a CUDA model; capture=False is for diagnostics")
        self.prefixes = [[int(token) for token in prefix] for prefix in prefixes]
        self.max_tokens, self.branches = int(max_tokens), len(prefixes)
        self.capacity = max(map(len, prefixes)) + self.max_tokens
        self.capture, self.graph, self.output = capture, None, None
        if attention_backend not in {"auto", "flash", "cudnn", "sdpa"}:
            raise ValueError("attention_backend must be auto, flash, cudnn, or sdpa")
        fused = self.device.type == "cuda" and self.dtype in {torch.bfloat16, torch.float16} and config.head_dim % 8 == 0
        flash = fused and config.head_dim <= 256 and hasattr(torch.ops.aten, "_flash_attention_forward") and (
            "seqused_k" in str(torch.ops.aten._flash_attention_forward.default._schema))
        # Torch 2.10 is pinned by the package. Its native variable-length FA
        # accepts GPU effective lengths; the public masked SDPA can select a
        # much slower math kernel. Keep a cuDNN/public-SDPA fallback explicit.
        if attention_backend == "auto":
            attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"
        if attention_backend == "flash" and not flash:
            raise ValueError("Pinned PyTorch variable-length CUDA FlashAttention is unavailable")
        if attention_backend == "cudnn" and not (fused and torch.backends.cudnn.is_available()):
            raise ValueError("cuDNN attention requires a supported CUDA dtype/head dimension")
        self.attention_backend = attention_backend
        self.ready, self.closed, self.steps = False, False, 0
        # Sequence-major layout makes the packed FA view contiguous without
        # copying all cached keys at each decode step.
        shape = (self.branches, self.capacity, config.num_key_value_heads, config.head_dim)
        self.keys = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in model.model.layers]
        self.values = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in model.model.layers]
        self.positions = torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.long, device=self.device)
        self.initial_positions = self.positions.clone()
        self.key_positions = torch.arange(self.capacity, dtype=torch.long, device=self.device)
        self.cu_q = torch.arange(self.branches + 1, dtype=torch.int32, device=self.device)
        self.cu_k = self.cu_q * self.capacity
        self.tokens = torch.tensor([[prefix[-1]] for prefix in prefixes], dtype=torch.long, device=self.device)
        self.kv_bytes = 2 * len(self.keys) * self.keys[0].numel() * self.keys[0].element_size()
        self.fused_weights = []
        if fuse_projections:
            if any(module.bias is not None for module in linears):
                raise ValueError("Fused projections require the checkpoint's bias-free AR linears")
            with torch.no_grad():
                for layer in model.model.layers:
                    self.fused_weights.append((torch.cat([layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
                                                         layer.self_attn.v_proj.weight], dim=0),
                                               torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0)))
        self.fused_weight_bytes = sum(value.numel() * value.element_size() for pair in self.fused_weights for value in pair)
        self.use_fused_kernels = False
        if _fused_enabled() and self.device.type == "cuda" and \
                self.dtype in {torch.bfloat16, torch.float16} and not fuse_projections:
            try:
                import sgl_kernel  # noqa: F401

                self.use_fused_kernels = True
            except ImportError:
                self.use_fused_kernels = False
        self.positions_i32 = self.positions.to(torch.int32)
        self._fused_layer_weights = model_fused_weights(model) if self.use_fused_kernels else None
        self.sampler = sampler
        self.step_mode = sampler is not None

    @torch.inference_mode()
    def _decode(self):
        if self.use_fused_kernels:
            return self._decode_fused()
        backbone = self.model.model
        cos, sin = backbone.rotary_emb(self.positions[:, None])
        x = backbone.embed_tokens(self.tokens)
        visible = None
        if self.attention_backend != "flash":
            visible = (self.key_positions[None, :] <= self.positions[:, None])[:, None, None, :]
        used_lengths = (self.positions + 1).to(torch.int32)
        config = self.model.config
        slots = self.positions[:, None, None, None].expand(
            self.branches, 1, config.num_key_value_heads, config.head_dim)
        for index, (layer, keys, values) in enumerate(zip(backbone.layers, self.keys, self.values)):
            normalized = layer.input_layernorm(x)
            if self.fused_weights:
                from .modeling_yue2 import _apply_rotary
                sizes = (config.num_attention_heads * config.head_dim,
                         config.num_key_value_heads * config.head_dim, config.num_key_value_heads * config.head_dim)
                q, k, v = F.linear(normalized, self.fused_weights[index][0]).split(sizes, dim=-1)
                q = layer.self_attn.q_norm(q.view(self.branches, 1, config.num_attention_heads, config.head_dim))
                k = layer.self_attn.k_norm(k.view(self.branches, 1, config.num_key_value_heads, config.head_dim))
                v = v.view(self.branches, 1, config.num_key_value_heads, config.head_dim)
                q = _apply_rotary(q, cos.unsqueeze(2), sin.unsqueeze(2))
                k = _apply_rotary(k, cos.unsqueeze(2), sin.unsqueeze(2))
            else:
                q, k, v = layer.self_attn.project_qkv(normalized, cos, sin)
            keys.scatter_(1, slots, k)
            values.scatter_(1, slots, v)
            # All allocated slots are present, but only each branch's completed
            # prefix and the current token are visible. Future slots never leak.
            if self.attention_backend == "flash":
                # seqused_k is respected by the 3D packed/varlen entrypoint.
                # The 4D fixed-batch entrypoint ignores it in torch 2.10, so do
                # not replace this call with an apparently equivalent 4D call.
                h = torch.ops.aten._flash_attention_forward(
                    q[:, 0], keys.view(-1, config.num_key_value_heads, config.head_dim),
                    values.view(-1, config.num_key_value_heads, config.head_dim),
                    self.cu_q, self.cu_k, 1, self.capacity, 0.0, False, False,
                    seqused_k=used_lengths)[0][:, None]
            elif self.attention_backend == "cudnn":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    h = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                            attn_mask=visible, is_causal=False,
                            enable_gqa=config.num_attention_heads != config.num_key_value_heads).transpose(1, 2)
            else:
                h = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                            attn_mask=visible, is_causal=False,
                            enable_gqa=config.num_attention_heads != config.num_key_value_heads).transpose(1, 2)
            x = x + layer.self_attn.o_proj(h.reshape(self.branches, 1, -1))
            normalized = layer.post_attention_layernorm(x)
            if self.fused_weights:
                gate, up = F.linear(normalized, self.fused_weights[index][1]).chunk(2, dim=-1)
                x = x + layer.mlp.down_proj(F.silu(gate) * up)
            else:
                x = x + layer.mlp(normalized)
        output = self.model.lm_head(backbone.norm(x))[:, 0]
        self.positions.add_(1)
        return output

    @torch.inference_mode()
    def _decode_fused(self):
        """Decode with SGLang fused kernels: fused_add_rmsnorm, fused QKV +
        gate/up GEMMs, fused per-head q/k RMSNorm + NeoX RoPE, fused SiLU-mul.

        Same math as the reference path; the residual stream lives in
        ``residual`` and every layer entry is pre-normed into ``hidden``.
        """
        import sgl_kernel

        backbone = self.model.model
        config = self.model.config
        B, H = self.branches, config.hidden_size
        HQ, HKV, HD = (config.num_attention_heads, config.num_key_value_heads,
                       config.head_dim)
        eps = config.rms_norm_eps
        residual = backbone.embed_tokens(self.tokens)
        hidden = sgl_kernel.rmsnorm(
            residual.view(B, H), backbone.layers[0].input_layernorm.weight, eps
        ).view(B, 1, H)
        visible = None
        if self.attention_backend != "flash":
            visible = (self.key_positions[None, :] <= self.positions[:, None])[:, None, None, :]
        used_lengths = (self.positions + 1).to(torch.int32)
        slots = self.positions[:, None, None, None].expand(B, 1, HKV, HD)
        last = len(backbone.layers) - 1
        for index, (layer, keys, values) in enumerate(zip(backbone.layers, self.keys, self.values)):
            qkv_w, gate_up_w = self._fused_layer_weights[index]
            qkv = F.linear(hidden.view(B, H), qkv_w)
            sgl_kernel.fused_qk_norm_rope(
                qkv, HQ, HKV, HKV, HD, eps,
                layer.self_attn.q_norm.weight, layer.self_attn.k_norm.weight,
                config.rope_theta, True, self.positions_i32, 1.0, 0.0, 0.0, 1.0,
            )
            q = qkv[:, : HQ * HD].view(B, 1, HQ, HD)
            k = qkv[:, HQ * HD: (HQ + HKV) * HD].view(B, 1, HKV, HD)
            v = qkv[:, (HQ + HKV) * HD:].view(B, 1, HKV, HD)
            keys.scatter_(1, slots, k)
            values.scatter_(1, slots, v)
            if self.attention_backend == "flash":
                h = torch.ops.aten._flash_attention_forward(
                    q[:, 0], keys.view(-1, HKV, HD), values.view(-1, HKV, HD),
                    self.cu_q, self.cu_k, 1, self.capacity, 0.0, False, False,
                    seqused_k=used_lengths)[0][:, None]
            elif self.attention_backend == "cudnn":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    h = F.scaled_dot_product_attention(
                        q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                        attn_mask=visible, is_causal=False, enable_gqa=HQ != HKV,
                    ).transpose(1, 2)
            else:
                h = F.scaled_dot_product_attention(
                    q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                    attn_mask=visible, is_causal=False, enable_gqa=HQ != HKV,
                ).transpose(1, 2)
            attn = layer.self_attn.o_proj(h.reshape(B, 1, -1)).view(B, H)
            sgl_kernel.fused_add_rmsnorm(
                attn, residual.view(B, H), layer.post_attention_layernorm.weight, eps)
            gate_up = F.linear(attn, gate_up_w)
            mlp = layer.mlp.down_proj(sgl_kernel.silu_and_mul(gate_up))
            next_weight = (backbone.layers[index + 1].input_layernorm.weight
                           if index < last else backbone.norm.weight)
            sgl_kernel.fused_add_rmsnorm(mlp, residual.view(B, H), next_weight, eps)
            # fused_add_rmsnorm writes the normed stream into its first
            # argument, so after the final call ``mlp`` holds
            # rmsnorm(residual) * norm_weight — exactly what lm_head reads.
            hidden = mlp.view(B, 1, H)
        output = self.model.lm_head(hidden)[:, 0]
        self.positions.add_(1)
        self.positions_i32.add_(1)
        return output

    @torch.inference_mode()
    def _step_once(self):
        """Captured body for the fused decode+sample step graph.

        The sampled token feeds the next replay's decode entirely on the GPU;
        the sampler keeps its ring/step state in graph buffers.
        """
        self.tokens.copy_(self.sampler.token_buf.expand(self.branches, 1))
        logits = self._decode()
        token, done = self.sampler.sample_in_graph(logits)
        self.sampler.advance_in_graph()
        return token, done

    @torch.inference_mode()
    def _capture(self):
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            warmup = torch.cuda.Stream(device=self.device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    self.positions.copy_(self.initial_positions)
                    if self.step_mode:
                        self._step_once()
                    else:
                        self._decode()
                self.positions.copy_(self.initial_positions)
                if self.step_mode:
                    self.sampler.reset_state()
            current.wait_stream(warmup)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            if self.step_mode:
                # The in-graph multinomial draws from this generator; torch
                # requires its RNG state registered before capture_begin.
                self.graph.register_generator_state(self.sampler.generator)
            with torch.cuda.graph(self.graph):
                if self.step_mode:
                    self.output = self._step_once()
                else:
                    self.output = self._decode()
            # Warmup/capture wrote one future slot. The first real step
            # overwrites that slot in every layer before making it visible.
            self.positions.copy_(self.initial_positions)
            self.positions_i32.copy_(self.initial_positions)
            if self.step_mode:
                self.sampler.reset_state()

    @torch.inference_mode()
    def replay_step(self):
        """One fused decode+sample replay; returns (token_buf, done_buf)."""
        if self.closed or not self.ready:
            raise RuntimeError("Call prefill before replay_step")
        if self.graph is None or not self.step_mode:
            raise RuntimeError("replay_step requires a captured step graph")
        self.graph.replay()
        return self.sampler.token_buf, self.sampler.done_buf

    @torch.inference_mode()
    def step_eager(self, token):
        """Decode one token without the captured graph (KV building for NAR)."""
        if self.closed or not self.ready:
            raise RuntimeError("Call prefill before step_eager")
        if isinstance(token, Integral) and not isinstance(token, bool):
            if not 0 <= int(token) < self.model.config.vocab_size:
                raise ValueError("Token is outside the model vocabulary")
            self.tokens.fill_(int(token))
        else:
            self.tokens.copy_(torch.tensor([[int(token)]], device=self.device,
                                           dtype=torch.long))
        result = self._decode()
        self.steps += 1
        return result

    @torch.inference_mode()
    def prefill(self):
        if self.closed or self.ready:
            raise RuntimeError("prefill must be called exactly once on an open GraphAR")
        logits = []
        for branch, prefix in enumerate(self.prefixes):
            cache = _PrefixCache(self.keys, self.values, branch)
            result = self.model(torch.tensor([prefix], dtype=torch.long, device=self.device),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits.append(result.logits[:, -1])
        result = torch.cat(logits, dim=0)
        if self.capture and self.max_tokens > 1 and self.graph is None:
            self._capture()
        self.ready = True
        return result

    @torch.inference_mode()
    def step(self, token):
        if self.step_mode:
            # The captured graph chains decode+sample from token_buf; an
            # explicit token decode must go through the eager path.
            return self.step_eager(token)
        if self.closed or not self.ready:
            raise RuntimeError("Call prefill before step and do not use a closed GraphAR")
        if self.steps >= self.max_tokens - 1:
            raise ValueError("Requested generation budget is exhausted")
        if isinstance(token, Integral) and not isinstance(token, bool):
            if not 0 <= int(token) < self.model.config.vocab_size:
                raise ValueError("Token is outside the model vocabulary")
            self.tokens.fill_(int(token))
        elif isinstance(token, torch.Tensor) and token.numel() == 1 and token.dtype in {torch.int32, torch.int64}:
            # Sampling already produced/validated this scalar. Copying the
            # tensor avoids a second GPU-to-CPU synchronization in the caller.
            if token.device.type == "cpu" and not 0 <= token.item() < self.model.config.vocab_size:
                raise ValueError("Token is outside the model vocabulary")
            self.tokens.copy_(token.reshape(1, 1).expand(self.branches, 1))
        else:
            raise ValueError("step needs one shared integer token, including for CFG")
        if self.graph is not None:
            self.graph.replay()
            result = self.output
        else:
            result = self._decode()
        self.steps += 1
        return result

    def close(self):
        self.graph = self.output = None
        self.keys.clear()
        self.values.clear()
        self.fused_weights.clear()
        self._fused_layer_weights = None
        self.positions_i32 = None
        self.tokens = self.positions = self.initial_positions = self.key_positions = None
        self.cu_q = self.cu_k = None
        self.closed = True

    def reopen(self, prefixes, max_tokens):
        """Reset a pooled session for a new request without recapturing.

        The captured graph topology only depends on (branches, capacity), so
        reusing a session only needs fresh buffers: the prefix tail token,
        RoPE positions, and the per-request step budget. Stale KV slots beyond
        the new prefix are never read: flash honours ``seqused_k`` and the
        masked backends apply the ``visible`` mask, and prefill overwrites
        slots ``0..len(prefix)-1`` while the first decode overwrites the next
        slot before it becomes visible.
        """
        if self.closed:
            raise RuntimeError("reopen requires an open session")
        if len(prefixes) != self.branches:
            raise ValueError("Branch count changed for a pooled session")
        needed = max(map(len, prefixes)) + max_tokens
        if needed > self.capacity:
            raise ValueError("Pooled session capacity exceeded")
        self.prefixes = [[int(token) for token in prefix] for prefix in prefixes]
        self.max_tokens = int(max_tokens)
        self.steps = 0
        self.ready = False
        with torch.no_grad():
            self.tokens.copy_(torch.tensor([[prefix[-1]] for prefix in prefixes],
                                           dtype=torch.long, device=self.device))
            self.positions.copy_(torch.tensor([len(prefix) for prefix in prefixes],
                                              dtype=torch.long, device=self.device))
            self.initial_positions.copy_(self.positions)
            self.positions_i32.copy_(self.positions)


class GraphSessionPool:
    """Process-wide pool of captured AR decode graphs, keyed by (branches, capacity).

    Capturing one decode graph costs ~35 ms; the pool amortizes it across
    requests. Sessions are borrowed for the whole request (AR phases and the
    NAR stage's KV reuse) and returned via :meth:`release`.
    """

    def __init__(self, bucket: int = 256, max_sessions_per_key: int = 4):
        self.bucket = int(bucket)
        self.max_sessions_per_key = int(max_sessions_per_key)
        self._idle: dict[tuple[int, int], list] = {}
        self._live = 0
        self._lock = threading.Lock()

    def capacity_for(self, prefixes, max_tokens) -> int:
        need = max(map(len, prefixes)) + int(max_tokens)
        return -(-need // self.bucket) * self.bucket

    def acquire(self, model, prefixes, max_tokens):
        prefixes = [list(prefix) for prefix in prefixes]
        capacity = self.capacity_for(prefixes, max_tokens)
        # GraphAR sizes capacity as max(len(prefix)) + max_tokens, so the
        # per-instance budget is the bucket minus the longest prefix.
        budget = capacity - max(map(len, prefixes))
        key = (len(prefixes), capacity)
        session = None
        with self._lock:
            idle = self._idle.get(key)
            if idle:
                session = idle.pop()
        if session is not None:
            session.reopen(prefixes, budget)
            return session
        session = GraphAR(model, prefixes, budget)
        with self._lock:
            self._live += 1
        return session

    def release(self, session):
        if session is None or session.closed:
            return
        key = (session.branches, session.capacity)
        with self._lock:
            idle = self._idle.setdefault(key, [])
            if len(idle) < self.max_sessions_per_key:
                idle.append(session)
                return
        session.close()

    def stats(self) -> dict:
        with self._lock:
            return {"live_sessions": self._live,
                    "idle": {str(k): len(v) for k, v in self._idle.items()}}


default_session_pool = GraphSessionPool()
