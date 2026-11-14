"""Experimental Qwen3.5 TP weight prefetching, opt-in via sitecustomize.

Schedule read-only weight prefetches on a side CUDA stream around AllReduce.
The default 16/32 MiB budgets and lazy stream join preserve the original
experiment. Lazy mode permits overlap with subsequent GEMMs; it does not
confine prefetch traffic to the collective.

Bulk prefetch is a cache hint, not a residency or memory-completion guarantee.
It may contend for memory bandwidth and L2 capacity. Read-only weights do not
establish bitwise-identical end-to-end generation. See README.md for the tested
configuration, reproduction steps, and limitations.
"""

import os

import torch
import triton
import triton.language as tl

MAX_TOKENS = int(os.environ.get("SGLANG_L2PF_MAX_TOKENS", 512))
SMALL_TOKENS = int(os.environ.get("SGLANG_L2PF_SMALL_TOKENS", 16))
BUDGETS = (
    int(float(os.environ.get("SGLANG_L2PF_BYTES_SMALL", 16 * 2**20))),
    int(float(os.environ.get("SGLANG_L2PF_BYTES", 32 * 2**20))),
)
KIND = os.environ.get("SGLANG_L2PF_KIND", "bulk")
JOIN = os.environ.get("SGLANG_L2PF_JOIN", "lazy")
CTAS = 170

_st = {
    "next": None,
    "side": None,
    "join": None,
    "sink": None,
    "stats": {"pf": 0, "skip": 0},
}


@triton.jit
def _pf_load(ptr, n_words, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(pid * BLOCK, n_words, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(
            ptr + offs, mask=offs < n_words, other=0, eviction_policy="evict_last"
        )
    tl.store(out + pid, tl.sum(acc, 0))


@triton.jit
def _pf_bulk(base, nbytes, out, CHUNK: tl.constexpr, LANES: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * LANES + tl.arange(0, LANES)
    off = idx.to(tl.int64) * CHUNK
    valid = off < nbytes
    size = tl.where(valid, tl.minimum(nbytes - off, CHUNK), 16).to(tl.int32)
    addr = base + tl.where(valid, off, 0)
    r = tl.inline_asm_elementwise(
        "cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;",
        "=r,l,r",
        [addr, size],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.store(out + pid, tl.sum(r, 0))


def _chunks(modules, budget):
    """Collect contiguous CUDA weight views in module order, up to budget."""
    seen, out, total = set(), [], 0
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            if not t.is_cuda or not t.is_contiguous() or t.data_ptr() in seen:
                continue
            nbytes = t.numel() * t.element_size() // 128 * 128
            if nbytes < 64 * 1024 or total >= budget:
                continue
            seen.add(t.data_ptr())
            n = min(nbytes, budget - total)
            out.append(t.detach().reshape(-1).view(torch.uint8)[:n].view(torch.int32))
            total += n
    return out


def _prefetch(views):
    sink = _st["sink"]
    for v in views:
        if KIND == "bulk":
            nbytes = v.numel() * 4
            _pf_bulk[(triton.cdiv(triton.cdiv(nbytes, 65536), 32),)](
                v.data_ptr(), nbytes, sink, CHUNK=65536, LANES=32, num_warps=1
            )
        else:
            _pf_load[(min(CTAS, triton.cdiv(v.numel(), 2048)),)](
                v, v.numel(), sink, BLOCK=2048, num_warps=4
            )


def _drain():
    ev = _st["join"]
    if ev is not None:
        torch.cuda.current_stream().wait_event(ev)
        _st["join"] = None


# Intercept AllReduce.
def _patched_all_reduce(orig):
    def all_reduce(self, input_):
        plans, _st["next"] = _st["next"], None
        if JOIN == "lazy":
            _drain()
        if (
            not plans
            or self.world_size == 1
            or input_.dim() == 0
            or input_.shape[0] > MAX_TOKENS
        ):
            if plans:
                _st["stats"]["skip"] += 1
            return orig(self, input_)
        views = plans[0] if input_.shape[0] <= SMALL_TOKENS else plans[1]
        cur, side = torch.cuda.current_stream(), _st["side"]
        fork, join = torch.cuda.Event(), torch.cuda.Event()
        fork.record(cur)
        with torch.cuda.stream(side):
            side.wait_event(fork)
            _prefetch(views)
            join.record(side)
        out = orig(self, input_)
        if JOIN == "lazy":
            _st["join"] = join
        else:
            cur.wait_event(join)
        _st["stats"]["pf"] += 1
        return out

    return all_reduce


# Register the weights expected after each AllReduce.
def _layer_parts(layer):
    if hasattr(layer, "linear_attn"):
        la = layer.linear_attn
        return [la.in_proj_qkvz, la.in_proj_ba], la.out_proj
    att = getattr(layer, "self_attn", layer)
    return [att.qkv_proj], att.o_proj


def _arm(model):
    layers = [model.layers[i] for i in range(model.start_layer, model.end_layer)]
    parts = [_layer_parts(l) for l in layers]
    plan = []  # (AllReduce producer, subsequent GEMM modules)
    if hasattr(model, "embed_tokens"):
        plan.append((model.embed_tokens, parts[0][0] + [parts[0][1]]))
    for i, (layer, (_ins, outp)) in enumerate(zip(layers, parts)):
        plan.append((outp, [layer.mlp.gate_up_proj, layer.mlp.down_proj]))
        if i + 1 < len(layers):
            nxt_in, nxt_out = parts[i + 1]
            plan.append((layer.mlp.down_proj, nxt_in + [nxt_out]))
    for producer, consumers in plan:
        plans = tuple(_chunks(consumers, b) for b in BUDGETS)

        def hook(_mod, _args, plans=plans):
            _st["next"] = plans

        producer.register_forward_pre_hook(hook)
    return len(plan)


def install():
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models import qwen3_5

    GroupCoordinator.all_reduce = _patched_all_reduce(GroupCoordinator.all_reduce)

    orig_init = ModelRunner.init_cuda_graphs

    def init_cuda_graphs(self, *a, **k):
        if not self.is_draft_worker and not getattr(self, "_l2pf_armed", False):
            _st["side"] = torch.cuda.Stream()
            _st["sink"] = torch.zeros(4096, dtype=torch.int32, device="cuda")
            inner = next(
                m
                for m in self.model.modules()
                if isinstance(m, qwen3_5.Qwen3_5ForCausalLM)
            )
            n = _arm(inner)
            self._l2pf_armed = True
            print(
                f"L2PF armed: {n} AR sites, budgets={[b >> 20 for b in BUDGETS]}MiB "
                f"small_tokens={SMALL_TOKENS} kind={KIND} join={JOIN} max_tokens={MAX_TOKENS}",
                flush=True,
            )
        out = orig_init(self, *a, **k)
        if not self.is_draft_worker:
            print(f"L2PF after graph capture: {_st['stats']}", flush=True)
        return out

    ModelRunner.init_cuda_graphs = init_cuda_graphs

    orig_fwd = qwen3_5.Qwen3_5ForCausalLM.forward

    def forward(self, *a, **k):
        try:
            return orig_fwd(self, *a, **k)
        finally:
            _st["next"] = None
            _drain()

    qwen3_5.Qwen3_5ForCausalLM.forward = forward
    print("L2PF installed", flush=True)
