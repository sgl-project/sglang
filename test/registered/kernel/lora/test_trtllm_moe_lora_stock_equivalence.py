"""Layer-level equivalence for the stock-FlashInfer BF16 MoE-LoRA dispatch.

``fused_experts_none_to_experimental_sgl_trtllm_bf16_lora`` runs stock
``flashinfer.fused_moe.trtllm_bf16_routed_moe`` with the gate_up LoRA delta handed
over as ``gemm1_lora_delta``, gathers the returned permuted post-SwiGLU activation
back into expanded ``[num_tokens, top_k, inter]`` order, and merges the
virtual-experts down-LoRA into the finalized output. Four things there can be
wrong without looking wrong, and each gets its own assertion:

- **the half swap.** trtllm-gen adds ``gemm1_lora_delta[..., :inter]`` to FC1's
  FIRST half, which the trtllm w13 prep loads as ``up``, while the LoRA expand
  natively emits ``[gate | up]``. Crossing the pairing produces perfectly
  plausible numbers, so it is checked twice: bit-exactly on the delta that
  reaches the op, and numerically with a gate-only adapter whose reference moves
  the opposite way if the halves are not swapped.
- **the gather.** The op's activation comes back permuted; the down-LoRA reads it
  expanded and sums over every slot unconditionally, so inactive slots must read
  back as exact zeros. Checked three ways: against a torch gather of the op's own
  tensor, against a torch model of the post-SwiGLU FC1 (so a zeroed or
  pre-activation return slot cannot pass), and through the consumer -- poisoning
  the inactive rows must make the merged down-LoRA non-finite.
- **the numerics end to end**, against a torch reference of the same math, with a
  per-row max metric alongside the Frobenius one: localized corruption (a dropped
  stream join, a slot never written) disappears into a whole-tensor norm.
- **the tokens with no adapter**, which must contribute exactly zero LoRA.

The two shape guards the dispatch asserts up front (BlockMajorK w13, no fused
shared experts) get a test each; they are pure Python and fire before any launch.

Rank > 64 is covered structurally only. It takes the generic expand kernel, which
derives its K from ``lora_b.shape[2] == rank`` and therefore contracts BOTH output
halves against the gate-shrink columns ``A[0:rank]`` -- a pre-existing gated-split
defect that the rank-specialized kernel's ``GATED_A_HALF`` fixes and this
migration does not touch. Comparing that rank against a correct PEFT reference
would be asserting the old bug, so only the swap itself is checked there.

Usage:
    python -m pytest test/registered/kernel/lora/test_trtllm_moe_lora_stock_equivalence.py -v
"""

import contextlib
import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

# Every test here drives flashinfer.fused_moe.trtllm_bf16_routed_moe, which resolves
# through get_trtllm_moe_sm100_module(). That is a build failure, not a numeric one,
# off SM100 -- so this registers to the Blackwell job (pr-test-jit-kernel.yml runs
# base-b-kernel-unit-test-4-gpu-b200) rather than the 1-gpu-large H100 one.
register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _unsupported_reason() -> str:
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability()[0] != 10:
        return "trtllm-gen BF16 routed MoE is SM100-only"
    try:
        from flashinfer.fused_moe import trtllm_bf16_routed_moe
    except ImportError:
        return "flashinfer.fused_moe is required"
    # gemm1_lora_delta landed in FlashInfer #3153; an older image raises a
    # TypeError deep inside the call instead of anything readable.
    if "gemm1_lora_delta" not in inspect.signature(trtllm_bf16_routed_moe).parameters:
        return "installed FlashInfer has no gemm1_lora_delta"
    return ""


_UNSUPPORTED = _unsupported_reason()

pytestmark = pytest.mark.skipif(bool(_UNSUPPORTED), reason=_UNSUPPORTED or "supported")

DEV = "cuda"
DTYPE = torch.bfloat16

# Inkling at TP=8 is H=6144, I=384, top_k=6, E=256. inter and top_k are kept at the
# real values -- inter drives the gate/up tile alignment the swap depends on -- while
# hidden and the expert count are cut to the smallest shapes the trtllm-gen BF16
# kernels accept (hidden % 128 == 0; upstream's own delta test runs hidden=1024).
HIDDEN = 1024
INTER = 384
TOP_K = 6
NUM_EXPERTS = 32
MAX_LORAS = 2


def _runtime_scaffolding():
    """The dispatch needs global server args and a TP group.

    It allocates its output under ``use_symmetric_memory(get_tp_group(), ...)``
    even when symmetric allocation is off, and the LoRA triton config lookup reads
    server args. Single rank, gloo, TP=EP=PP=1.
    """
    import os

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29653")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


@pytest.fixture(scope="module", autouse=True)
def _scaffolding():
    _runtime_scaffolding()


def _prepare_trtllm_bf16_weights(
    w13_logical: torch.Tensor, w2_logical: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ``unquant.py``'s flashinfer_trtllm weight prep, expert by expert.

    Yields the 4-D BlockMajorK tensors the routed BF16 entry point requires:
    reorder for the gated activation, shuffle for the epilogue tile, then block
    the K dimension.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    cache: dict = {}
    epilogue_tile_m, block_k = 128, 128
    w13_blocks, w2_blocks = [], []
    for e in range(w13_logical.shape[0]):
        idx13 = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            w13_logical[e].view(torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=True,
        )
        w13_e = w13_logical[e].view(torch.uint8)[idx13.to(DEV)].contiguous()
        w13_blocks.append(convert_to_block_layout(w13_e, block_k).view(DTYPE))

        idx2 = get_w2_permute_indices_with_cache(
            cache, w2_logical[e].view(torch.uint8), epilogue_tile_m
        )
        w2_e = w2_logical[e].view(torch.uint8)[idx2.to(DEV)].contiguous()
        w2_blocks.append(convert_to_block_layout(w2_e, block_k).view(DTYPE))

    return torch.stack(w13_blocks).contiguous(), torch.stack(w2_blocks).contiguous()


def _build_case(rank: int, num_tokens: int, seed: int = 0):
    """One Inkling-shaped BF16 MoE layer plus a 2-adapter gate_up/down LoRA."""
    torch.manual_seed(seed)

    def randn(*shape):
        return (torch.randn(*shape, device=DEV, dtype=torch.float32) * 0.02).to(DTYPE)

    hidden_states = randn(num_tokens, HIDDEN)
    # [up | gate], the order models/inkling.py loads w13 in for the trtllm backend.
    w13_logical = randn(NUM_EXPERTS, 2 * INTER, HIDDEN)
    w2_logical = randn(NUM_EXPERTS, HIDDEN, INTER)
    w13, w2 = _prepare_trtllm_bf16_weights(w13_logical, w2_logical)

    # gate_A must differ from up_A: with the two equal, contracting both output
    # halves against A[0:rank] is invisible.
    lora = SimpleNamespace(
        gate_up_a=randn(MAX_LORAS, NUM_EXPERTS, 2 * rank, HIDDEN),
        gate_up_b=randn(MAX_LORAS, NUM_EXPERTS, 2 * INTER, rank),
        down_a=randn(MAX_LORAS, NUM_EXPERTS, rank, INTER),
        down_b=randn(MAX_LORAS, NUM_EXPERTS, HIDDEN, rank),
    )

    topk_ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, device=DEV)[:TOP_K] for _ in range(num_tokens)]
    ).to(torch.int32)
    # Round the routing weights to bf16 up front: the packed topk the op consumes
    # carries them as bf16, so the torch reference would otherwise diverge on the
    # weights alone.
    topk_weights = (
        torch.softmax(torch.randn(num_tokens, TOP_K, device=DEV), dim=-1)
        .to(DTYPE)
        .float()
    )

    token_lora_mapping = torch.randint(
        0, MAX_LORAS, (num_tokens,), device=DEV, dtype=torch.int32
    )
    # Half the tokens carry no adapter -- the -1 sentinel path. Keep at least one
    # adapter token so a decode-sized batch still exercises the LoRA at all.
    token_lora_mapping[max(1, num_tokens // 2) :] = -1

    return SimpleNamespace(
        rank=rank,
        num_tokens=num_tokens,
        hidden_states=hidden_states,
        w13_logical=w13_logical,
        w2_logical=w2_logical,
        w13=w13,
        w2=w2,
        lora=lora,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
    )


def _moe_call_args(case):
    """The dispatch/quant-info/runner-config triple both paths below take."""
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmBf16MoeQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import RoutingMethodType

    dispatch_output = StandardDispatchOutput(
        hidden_states=case.hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            router_logits=None,
        ),
    )
    quant_info = FlashInferTrtllmBf16MoeQuantInfo(
        gemm1_weights=case.w13,
        gemm2_weights=case.w2,
        global_num_experts=NUM_EXPERTS,
        local_expert_offset=0,
    )
    runner_config = MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTER,
        top_k=TOP_K,
        num_fused_shared_experts=0,
        # Inkling's gate pre-routes, so the runner asks for TopK (no renormalize).
        routing_method_type=RoutingMethodType.TopK,
        activation="silu",
        is_gated=True,
    )
    return dispatch_output, quant_info, runner_config


def _lora_info(case, lora=None):
    """The ``LoRAInfo`` the dispatch reads: two adapters, per-expert weights."""
    from sglang.srt.lora.lora_moe_runners import LoRAInfo

    lora = lora if lora is not None else case.lora
    return LoRAInfo(
        gate_up_lora_a_weights=lora.gate_up_a,
        gate_up_lora_b_weights=lora.gate_up_b,
        down_lora_a_weights=lora.down_a,
        down_lora_b_weights=lora.down_b,
        seg_indptr=torch.tensor([0, case.num_tokens], dtype=torch.int32, device=DEV),
        req_to_lora=torch.zeros(1, dtype=torch.int32, device=DEV),
        lora_ranks=torch.full((MAX_LORAS,), case.rank, dtype=torch.int32, device=DEV),
        adapter_enabled=torch.ones(MAX_LORAS, dtype=torch.int32, device=DEV),
        token_lora_mapping=case.token_lora_mapping,
        max_lora_rank=case.rank,
        num_experts=NUM_EXPERTS,
        has_active_lora=True,
        lora_use_virtual_experts=True,
        hidden_size=HIDDEN,
    )


def _run_dispatch(case, lora=None):
    """Call the migrated BF16 LoRA dispatch and return its ``[T, H]`` output."""
    from sglang.srt.lora.trtllm_lora_temp.lora_dispatch import (
        fused_experts_none_to_experimental_sgl_trtllm_bf16_lora,
    )

    combine = fused_experts_none_to_experimental_sgl_trtllm_bf16_lora(
        *_moe_call_args(case), _lora_info(case, lora)
    )
    return combine.hidden_states


def _run_base(case):
    """Same layer and routing, no LoRA: the stock non-LoRA BF16 routed path."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        fused_experts_none_to_flashinfer_trtllm_bf16,
    )

    return fused_experts_none_to_flashinfer_trtllm_bf16(
        *_moe_call_args(case), use_routed_topk=True
    ).hidden_states


@contextlib.contextmanager
def _spy_on_dispatch():
    """Record the delta handed to the op and the tensors the gather moves.

    Both are looked up on their module at call time (the dispatch imports them
    inside the function body), so patching the module attribute is enough.
    """
    import flashinfer.fused_moe as fi

    import sglang.kernels.ops.moe.moe_gather_permuted as gather_mod

    record = SimpleNamespace()
    original_moe = fi.trtllm_bf16_routed_moe
    original_gather = gather_mod.gather_permuted_activation

    def moe_spy(**kwargs):
        record.delta = kwargs["gemm1_lora_delta"].clone()
        return original_moe(**kwargs)

    def gather_spy(activation_permuted, expanded_idx_to_permuted_idx, **kwargs):
        # Clone the source: it is MoE workspace memory and may be reused after
        # the dispatch returns.
        record.activation_permuted = activation_permuted.clone()
        record.perm = expanded_idx_to_permuted_idx.clone()
        out = original_gather(
            activation_permuted, expanded_idx_to_permuted_idx, **kwargs
        )
        record.gathered = out.clone()
        return out

    fi.trtllm_bf16_routed_moe = moe_spy
    gather_mod.gather_permuted_activation = gather_spy
    try:
        yield record
    finally:
        fi.trtllm_bf16_routed_moe = original_moe
        gather_mod.gather_permuted_activation = original_gather


@contextlib.contextmanager
def _deterministic_lora_shrink():
    """Force the LoRA-A shrink to a single K split for the duration.

    ``_moe_lora_shrink_splitk_kernel`` accumulates its split-K partials with a
    **bf16** ``tl.atomic_add`` (``virtual_experts.py``, ``sem="relaxed"``), so
    once more than two CTAs contribute to an output element the reduction is
    order-dependent and the bf16 rounding differs between launches. Measured on
    B200 at this fixture's rank 32 (SPLIT_K == 4): two calls of
    ``merged_experts_fused_moe_lora_add`` with *identical* arguments disagree on
    ~1.2e4 of 73728 delta elements by one ULP (3.05e-05). Rank 128 draws
    SPLIT_K == 2, where the two partials only need commutativity, and is
    reproducible -- so the flakiness is latent there rather than absent.

    That is pre-existing SGLang behaviour, byte-identical to main, and unrelated
    to the half swap. But the swap is a pure permutation of the store column, so
    the property under test is bit-exactness, and a bit-exact comparison between
    two separate pipeline invocations needs a reproducible reduction.
    ``SPLIT_K == 1`` stores instead of accumulating and delivers exactly that;
    the test asserts the control actually took, so it cannot pass vacuously.
    """
    import sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts as ve

    original = ve._get_moe_lora_shrink_split_k
    ve._get_moe_lora_shrink_split_k = lambda *_a, **_k: 1
    try:
        yield
    finally:
        ve._get_moe_lora_shrink_split_k = original


def _shrink_expand(x, a, b):
    """One LoRA branch, bf16-rounded where the kernels store bf16."""
    return ((x @ a.float().t()).to(DTYPE).float() @ b.float().t()).to(DTYPE).float()


def _reference_activation(case, lora=None):
    """Post-SwiGLU FC1 activation in expanded ``[T, top_k, inter]`` order, fp32.

    The op's third return slot is supposed to BE this tensor (in permuted order).
    Comparing the gather against a torch gather of that same tensor cannot tell a
    real activation from a zeroed or pre-activation buffer, so assert the values.
    """
    lora = lora if lora is not None else case.lora
    rank = case.rank
    hidden_f = case.hidden_states.float()
    act = torch.zeros(case.num_tokens, TOP_K, INTER, dtype=torch.float32, device=DEV)

    for e in range(NUM_EXPERTS):
        slots = (case.topk_ids == e).nonzero(as_tuple=False)
        if slots.numel() == 0:
            continue
        t_idx, j_idx = slots[:, 0], slots[:, 1]
        x = hidden_f[t_idx]
        gate_up = x @ case.w13_logical[e].float().t()
        up = gate_up[:, :INTER].clone()  # first half = Up
        gate = gate_up[:, INTER:].clone()  # second half = Gate
        adapters = case.token_lora_mapping[t_idx]

        for adapter in adapters.unique().tolist():
            if adapter < 0:
                continue
            m = adapters == adapter
            up[m] += _shrink_expand(
                x[m],
                lora.gate_up_a[adapter, e, rank:],
                lora.gate_up_b[adapter, e, INTER:],
            )
            gate[m] += _shrink_expand(
                x[m],
                lora.gate_up_a[adapter, e, :rank],
                lora.gate_up_b[adapter, e, :INTER],
            )
        act[t_idx, j_idx] = (torch.nn.functional.silu(gate) * up).to(DTYPE).float()
    return act


def _reference(case, lora=None, cross_paired=False):
    """Torch PEFT reference for the same layer, in fp32.

    Mirrors the kernels' rounding points so the comparison is about the math and
    not about intermediate precision: the LoRA shrink, the gate_up delta and the
    post-SwiGLU activation are all bf16-rounded exactly where the kernels store
    them in bf16.

    ``cross_paired`` pairs gate-delta with Up and up-delta with Gate -- what the
    migrated path would compute if the half swap were dropped.
    """
    lora = lora if lora is not None else case.lora
    rank = case.rank
    hidden_f = case.hidden_states.float()
    out = torch.zeros(case.num_tokens, HIDDEN, dtype=torch.float32, device=DEV)

    for e in range(NUM_EXPERTS):
        slots = (case.topk_ids == e).nonzero(as_tuple=False)
        if slots.numel() == 0:
            continue
        t_idx, j_idx = slots[:, 0], slots[:, 1]
        x = hidden_f[t_idx]
        gate_up = x @ case.w13_logical[e].float().t()
        up = gate_up[:, :INTER].clone()  # first half = Up
        gate = gate_up[:, INTER:].clone()  # second half = Gate
        adapters = case.token_lora_mapping[t_idx]

        for adapter in adapters.unique().tolist():
            if adapter < 0:
                continue
            m = adapters == adapter
            gate_d = _shrink_expand(
                x[m],
                lora.gate_up_a[adapter, e, :rank],
                lora.gate_up_b[adapter, e, :INTER],
            )
            up_d = _shrink_expand(
                x[m],
                lora.gate_up_a[adapter, e, rank:],
                lora.gate_up_b[adapter, e, INTER:],
            )
            if cross_paired:
                up[m] += gate_d
                gate[m] += up_d
            else:
                up[m] += up_d
                gate[m] += gate_d

        act = (torch.nn.functional.silu(gate) * up).to(DTYPE).float()
        y = act @ case.w2_logical[e].float().t()
        for adapter in adapters.unique().tolist():
            if adapter < 0:
                continue
            m = adapters == adapter
            y[m] += _shrink_expand(
                act[m], lora.down_a[adapter, e], lora.down_b[adapter, e]
            )
        out.index_add_(0, t_idx, y * case.topk_weights[t_idx, j_idx].unsqueeze(1))
    return out


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


def _max_rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst per-element error, scaled by its own row's RMS.

    A Frobenius ratio over ``[T, H]`` dilutes one fully corrupted element to
    ``1 / sqrt(T * H)`` (2.8e-3 at T=129, H=1024) and sails under ``REL_TOL``.
    The failures this migration risks -- a dropped stream join, a slot the gather
    never wrote -- are localized, not spread, so they need a max metric.
    """
    rms = ref.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    return ((got.float() - ref.float()).abs() / rms).max().item()


# Both sides run bf16 tensor cores with fp32 accumulation and the reference
# reproduces every bf16 store the kernels make, so the residual is accumulation
# order over K=1024 (FC1) / K=384 (FC2) plus the bf16 rounding of the final
# output: order 1e-3 relative. 2e-2 keeps an order of magnitude of headroom while
# still being an order of magnitude tighter than a crossed half pairing, which
# moves the output by O(1) -- see test_gate_only_delta_lands_on_gate.
REL_TOL = 2e-2
# Per-element bound for the same residual: one bf16 ULP is 2^-8 of the element,
# and the largest element of a [1024] row sits around 3.5 row-RMS, so a clean run
# lands near 1.5e-2. This is a derived bound, not a measurement -- every check
# that uses it prints its achieved value so the first B200 run can pin it.
MAX_REL_TOL = 1e-1


@pytest.mark.parametrize("rank", [32, 128], ids=["direct_expand", "generic_expand"])
def test_delta_reaches_the_op_with_halves_swapped(rank):
    """The op must receive [up_delta | gate_delta], whichever expand produced it.

    Bit-exact: the swap only moves stores (rank <= 64) or is a plain ``torch.cat``
    (rank > 64), so nothing is recomputed either way. The reference expand runs
    under ``_deterministic_lora_shrink`` because the shared LoRA-A shrink -- not
    the swap -- is otherwise irreproducible across two calls.
    """
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    case = _build_case(rank, num_tokens=16)

    def unswapped_delta():
        """The same expand, no swap, into a NaN-poisoned destination.

        Poisoned rather than ``new_empty``: the delta buffer is read in full by
        the op (every expanded slot claims a permuted row at EP=1), so an element
        the expand never stores is a real defect. Against ``new_empty`` it would
        be two different pieces of uninitialized memory and the comparison below
        would fail for a reason that has nothing to do with the swap.
        """
        out = torch.full(
            (case.num_tokens, TOP_K, 2 * INTER), float("nan"), dtype=DTYPE, device=DEV
        )
        merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=case.hidden_states,
            lora_a=case.lora.gate_up_a,
            lora_b=case.lora.gate_up_b,
            topk_ids=case.topk_ids,
            topk_weights=case.topk_weights,
            token_lora_mapping=case.token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=False,
            fuse_add_to_output=False,
            use_direct_expand_add=rank <= 64,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
        )
        return out

    with _deterministic_lora_shrink():
        with _spy_on_dispatch() as record:
            _run_dispatch(case)
        unswapped = unswapped_delta()
        control = unswapped_delta()

    assert torch.isfinite(unswapped).all(), (
        "the expand left part of the gate_up delta unwritten; the op reads all of it"
    )
    # The control has to hold or the bit-exact assertion below proves nothing.
    assert torch.equal(unswapped, control), (
        "the LoRA-A shrink is still not reproducible; _deterministic_lora_shrink "
        "no longer forces SPLIT_K == 1"
    )
    expected = torch.cat((unswapped[..., INTER:], unswapped[..., :INTER]), dim=-1)
    assert torch.equal(record.delta, expected)
    # Guard against a symmetric adapter making the swap unobservable.
    assert not torch.equal(record.delta, unswapped)


def test_gathered_activation_matches_torch_gather():
    """The gather must equal the torch gather of the op's permuted activation."""
    case = _build_case(rank=32, num_tokens=16)
    with _spy_on_dispatch() as record:
        _run_dispatch(case)

    perm = record.perm.to(torch.int64)
    valid = perm >= 0
    ref = torch.zeros(case.num_tokens * TOP_K, INTER, dtype=DTYPE, device=DEV)
    ref[valid] = record.activation_permuted[perm[valid]]
    assert torch.equal(record.gathered.view(-1, INTER), ref)

    # At EP=1 with no cuda-graph padding every slot is local, so the line above
    # cannot exercise the -1 zero-fill the down-LoRA depends on. Re-run the gather
    # on the op's own tensors with part of the map forced inactive.
    from sglang.kernels.ops.moe.moe_gather_permuted import gather_permuted_activation

    holed = record.perm.clone()
    holed[::3] = -1
    poisoned = torch.full_like(record.gathered, float("nan"))
    gather_permuted_activation(
        record.activation_permuted,
        holed,
        num_tokens=case.num_tokens,
        top_k=TOP_K,
        out=poisoned,
    )
    holed_rows = poisoned.view(-1, INTER)
    still_active = holed >= 0
    assert (holed_rows[~still_active] == 0).all()
    assert torch.equal(
        holed_rows[still_active], record.gathered.view(-1, INTER)[still_active]
    )

    # Slot 2 of the op's return must BE the post-SwiGLU FC1 activation, not merely
    # something the gather moved faithfully: an all-zero or pre-activation buffer
    # satisfies the torch-gather comparison above and every other test in this file.
    # Only the rows the map names: the buffer is padded to
    # max_num_padded_tokens_gemm1 and its tile-padding rows hold workspace garbage.
    assert record.activation_permuted[perm[valid]].abs().amax() > 0
    ref_act = _reference_activation(case).view(-1, INTER)
    got_act = record.gathered.view(-1, INTER)
    rel = _rel_err(got_act[valid], ref_act[valid])
    print(f"activation rel={rel:.2e}")
    assert rel < REL_TOL


# 1 is the decode shape the two-stream gate cared about, 129 straddles a routing
# tile and is not a power of two.
@pytest.mark.parametrize("num_tokens", [1, 16, 129])
def test_matches_torch_reference(num_tokens):
    """End-to-end numerics against a torch PEFT reference of the same math."""
    case = _build_case(rank=32, num_tokens=num_tokens)
    got = _run_dispatch(case)
    ref = _reference(case)
    rel, max_rel = _rel_err(got, ref), _max_rel_err(got, ref)
    print(f"{num_tokens=} rel={rel:.2e} max={max_rel:.2e}")
    assert rel < REL_TOL
    assert max_rel < MAX_REL_TOL


def test_gate_only_delta_lands_on_gate():
    """An adapter whose up half is zero must move Gate, not Up.

    This is the assertion that fails if the halves are not swapped: the delta would
    land on Up, and the output would follow the ``cross_paired`` reference instead.
    The delta is scaled up to the same order as the base gate_up projection so the
    two references are unambiguously far apart rather than marginally so.
    """
    case = _build_case(rank=32, num_tokens=16)
    gate_only = SimpleNamespace(
        gate_up_a=case.lora.gate_up_a.clone(),
        # Zero the up half of LoRA-B so only the gate half of the delta is non-zero.
        gate_up_b=torch.cat(
            (
                case.lora.gate_up_b[:, :, :INTER] * 8.0,
                torch.zeros_like(case.lora.gate_up_b[:, :, INTER:]),
            ),
            dim=2,
        ).contiguous(),
        # No down-LoRA: keep the only moving part the gate_up delta.
        down_a=torch.zeros_like(case.lora.down_a),
        down_b=torch.zeros_like(case.lora.down_b),
    )

    got = _run_dispatch(case, lora=gate_only)
    ref = _reference(case, lora=gate_only)
    ref_crossed = _reference(case, lora=gate_only, cross_paired=True)

    # The test means nothing unless the two pairings actually disagree.
    crossed_gap = _rel_err(ref_crossed, ref)
    assert crossed_gap > 10 * REL_TOL, (
        f"gate-only delta is too small to distinguish the pairings ({crossed_gap=})"
    )
    assert _rel_err(got, ref) < REL_TOL
    assert _rel_err(got, ref_crossed) > crossed_gap / 2


def test_tokens_without_an_adapter_contribute_no_lora():
    """Rows with ``token_lora_mapping == -1`` must be bit-identical across adapters.

    Their FC1 bias rows are zeroed by the expand and the down-LoRA's -1 bucket
    never fires, so nothing about the adapter weights can reach them.
    """
    case = _build_case(rank=32, num_tokens=16)
    other = SimpleNamespace(
        gate_up_a=case.lora.gate_up_a * -3.0,
        gate_up_b=case.lora.gate_up_b + 0.05,
        down_a=case.lora.down_a * 2.0,
        down_b=case.lora.down_b - 0.05,
    )
    got = _run_dispatch(case)
    got_other = _run_dispatch(case, lora=other)

    no_adapter = case.token_lora_mapping < 0
    assert no_adapter.any() and (~no_adapter).any()
    assert torch.equal(got[no_adapter], got_other[no_adapter])
    # Not vacuous: the adapter rows did move.
    assert not torch.equal(got[~no_adapter], got_other[~no_adapter])

    # And they equal the plain non-LoRA path to within the bias-epilogue kernel
    # variant's own rounding (the LoRA run adds a zero BiasType::Mn epilogue).
    base = _run_base(case)
    rel = _rel_err(got[no_adapter], base[no_adapter])
    max_rel = _max_rel_err(got[no_adapter], base[no_adapter])
    print(f"no-adapter vs base rel={rel:.2e} max={max_rel:.2e}")
    assert rel < REL_TOL
    # 8 rows x 1024: the Frobenius ratio alone lets ~8 corrupted elements through.
    assert max_rel < MAX_REL_TOL


def test_inactive_slots_are_read_so_the_zero_fill_is_load_bearing():
    """The down-LoRA reads inactive slots, which is why the gather must zero them.

    ``test_gathered_activation_matches_torch_gather`` asserts the zero-fill as a
    property of the gather. This asserts it as a contract with the consumer: poison
    the rows the map leaves inactive and the merged down-LoRA goes non-finite, so an
    "optimization" that skips writing them (leaving ``torch.empty`` garbage, which
    looks safe because nobody appears to read them) turns this red.
    """
    from sglang.kernels.ops.moe.moe_gather_permuted import gather_permuted_activation
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    case = _build_case(rank=32, num_tokens=16)
    with _spy_on_dispatch() as record:
        _run_dispatch(case)

    holed = record.perm.clone()
    holed[::3] = -1
    # The poison only proves anything if a holed slot belongs to a token that has an
    # adapter: the -1 adapter bucket never reaches the shrink at all.
    has_adapter = (case.token_lora_mapping >= 0).repeat_interleave(TOP_K)
    assert ((holed < 0) & has_adapter).any()

    def _down(activation):
        out = torch.zeros(case.num_tokens, HIDDEN, dtype=DTYPE, device=DEV)
        merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=activation.view(-1, INTER),
            lora_a=case.lora.down_a,
            lora_b=case.lora.down_b,
            topk_ids=case.topk_ids,
            topk_weights=case.topk_weights,
            token_lora_mapping=case.token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=False,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=True,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
        )
        return out

    zeroed = torch.empty_like(record.gathered)
    gather_permuted_activation(
        record.activation_permuted,
        holed,
        num_tokens=case.num_tokens,
        top_k=TOP_K,
        out=zeroed,
    )
    poisoned = zeroed.clone()
    poisoned.view(-1, INTER)[holed < 0] = float("nan")

    assert _down(zeroed).isfinite().all()
    assert not _down(poisoned).isfinite().all(), (
        "the down-LoRA never read the inactive slots, so the gather's zero-fill is "
        "untested rather than satisfied"
    )


def test_rejects_flat_gate_up_weights():
    """A flat [E, 2F, D] w13 must fail at the shape, not inside the GEMM.

    The decomposed overlay op accepted it; the stock routed entry point takes
    BlockMajorK only, so the dispatch rejects it up front.
    """
    case = _build_case(rank=32, num_tokens=4)
    case.w13 = case.w13_logical
    with pytest.raises(AssertionError, match="BlockMajorK"):
        _run_dispatch(case)


def test_rejects_fused_shared_experts():
    """``expanded_idx_to_permuted_idx`` grows by the shared expert's extra slots."""
    from sglang.srt.lora.trtllm_lora_temp.lora_dispatch import (
        fused_experts_none_to_experimental_sgl_trtllm_bf16_lora,
    )

    case = _build_case(rank=32, num_tokens=4)
    dispatch_output, quant_info, runner_config = _moe_call_args(case)
    runner_config.num_fused_shared_experts = 1
    with pytest.raises(AssertionError, match="Fused shared experts"):
        fused_experts_none_to_experimental_sgl_trtllm_bf16_lora(
            dispatch_output, quant_info, runner_config, _lora_info(case)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
