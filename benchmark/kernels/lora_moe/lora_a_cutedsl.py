"""CuTeDSL tensor-core LoRA-A arms (P5; the §21.3 funnel obligation).

The masked grouped GEMM (the S2 base-GEMM winner) is expert-batched over a
dense ``[G, m_max, K]`` row domain — it does not gather.  The LoRA-A sites
are gather-GEMMs over the pair domain, so the CuTeDSL arm is a COMPOSITE:

1. **dispatch**: ``fused_moe_dispatch_index`` over the VIRTUAL topk ids
   (the fused ``(adapter, factor-expert)`` key — the same id domain every
   aligned plan groups by) yields ``masked_m[V]`` and the pair→row map
   ``src2dst[pair] = v * m_max + offset``;
2. **stage**: gate/up gathers token rows with the production
   ``fill_gateup_input_triton_kernel``; down places its PAIR-major
   activation rows with one ``index_copy_`` through the same ``src2dst``;
3. **GEMM**: the compiled masked grouped kernel (direct schedule, the
   provider's exact ABI) over ``A=[V, m_max, K]``, ``B=[V, N, K]`` factor
   weights, ``C=[V, m_max, N]``;
4. **scatter**: ``bridge[pair] = C.view(-1, N)[src2dst[pair]]`` on valid
   pairs — invalid (base-sentinel) rows stay undefined, the stock-B
   contract.

Both sites' tile schedules come from ONE ``build_dual_stage_schedules``
launch (``n_gemm1 = 2R`` gate/up, ``n_gemm2 = R`` down) off the same
``masked_m`` — the §45 dual-ownership rule holds by construction.

**m_max is right-sized to ``max(masked_m)``** (host-read at plan build,
OUTSIDE any timed thunk).  The production preprocess pads m_max to ~T per
group, which is fine for E_local base groups but TB-scale at V = E×L
virtual groups under prefill.  The host sync this sizing needs is a real
production question (capacity bounds / expected_m margins) — deliberately
NOT answered here: the funnel only needs the arm to exist at its most
favorable metadata boundary, because a loss under favorable terms rejects
the family safely, while a win reopens the sizing question with numbers.

Bitwise stability: the dispatch atomics order rows WITHIN a group
nondeterministically, but each C row is an independent K-reduction and the
scatter inverts src2dst exactly, so bridge VALUES are placement-invariant;
FP32 accumulation matches the Triton arms' signal-gate class (gate-close,
not bitwise, vs grouped — same as every cross-family comparison).

Compile discipline: one compile per (device, config, V, N, K) shared
process-wide through the provider's ``_COMPILE_CACHE`` (§61.1), zero-tile
warmup at compile so module load never lands inside a capture.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    fill_gateup_input_triton_kernel,
    fused_moe_dispatch_index,
)
from sglang.srt.lora.sgl_lora.routing import RouteView

# Provider-proven tile policy (cutedsl_bf16.py): widths and their compiled
# availability per arch. SM90 has no validated narrow token tile.
TOKEN_WIDTHS_SM100 = (8, 64, 128)
TOKEN_WIDTHS_SM90 = (64, 128)
OUTPUT_WIDTH = 128
PERSISTENT_CLUSTERS = 128


class CutedslAConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Sweepable kernel geometry for one compiled A-site GEMM."""

    token_width: int
    output_width: int = OUTPUT_WIDTH
    persistent_clusters: int = PERSISTENT_CLUSTERS
    mma_inst_tile_k: int = 4
    occupancy: int = 1


def supported_token_widths(device: torch.device) -> tuple[int, ...]:
    major, _ = torch.cuda.get_device_capability(device)
    return TOKEN_WIDTHS_SM100 if major >= 10 else TOKEN_WIDTHS_SM90


def _compiled_masked_gemm(
    *,
    device: torch.device,
    config: CutedslAConfig,
    num_groups: int,
    n: int,
    k: int,
) -> Any:
    """Compile (or fetch) the masked grouped GEMM for one site geometry."""
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
        _COMPILE_CACHE,
    )
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
        MaskedGroupedGemmConfig,
        prepare,
    )

    gemm_config = MaskedGroupedGemmConfig(
        mma_tiler_mn=(config.output_width, config.token_width),
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
        occupancy=config.occupancy,
        mma_inst_tile_k=config.mma_inst_tile_k,
        persistent_clusters=config.persistent_clusters,
        swap_ab=True,
        direct_schedule=True,
    )
    key = (device.type, device.index, gemm_config, num_groups, n, k, torch.bfloat16)
    compiled_fn = _COMPILE_CACHE.get(key)
    if compiled_fn is None:
        dummy_a = torch.zeros((num_groups, 256, k), dtype=torch.bfloat16, device=device)
        dummy_b = torch.zeros((num_groups, n, k), dtype=torch.bfloat16, device=device)
        dummy_c = torch.empty((num_groups, 256, n), dtype=torch.bfloat16, device=device)
        dummy_masked = torch.zeros(num_groups, dtype=torch.int32, device=device)
        prepared = prepare(dummy_a, dummy_b, dummy_c, dummy_masked, config=gemm_config)
        # Zero-tile warmup loads the CUDA module now, never inside a capture.
        prepared.launch()
        compiled_fn = prepared.compiled_fn
        _COMPILE_CACHE[key] = compiled_fn
    return compiled_fn


class CutedslLoraAPlan(msgspec.Struct, kw_only=True):
    """Per-fixture prepared state for BOTH A sites over one virtual route.

    Everything here is derivable per batch/layer; what a timed thunk
    re-executes is declared by the caller through the ``*_in_thunk``
    entry points below.
    """

    config: CutedslAConfig
    num_groups: int
    m_max: int
    top_k: int
    hidden_size: int
    act_size: int
    n_gate: int
    n_down: int
    virtual_topk_ids: torch.Tensor  # [T, K] int32, -1 sentinel
    valid_pair_ids: torch.Tensor  # [P_valid] int64
    masked_m: torch.Tensor  # [V] int32
    src2dst: torch.Tensor  # [T*K] int32 (valid positions only)
    src2dst_valid: torch.Tensor  # [P_valid] int64 gather/scatter rows
    staged_gate: torch.Tensor  # [V, m_max, H] bf16
    staged_down: torch.Tensor  # [V, m_max, I] bf16
    c_gate: torch.Tensor  # [V, m_max, 2R] bf16
    c_down: torch.Tensor  # [V, m_max, R] bf16
    schedule_gate: torch.Tensor
    tiles_gate: torch.Tensor
    schedule_down: torch.Tensor
    tiles_down: torch.Tensor
    compiled_gate: Any
    compiled_down: Any
    b_arg_gate: Any
    b_arg_down: Any
    # Torch owners of the wrapped weights (seventh S3 review): the DLPack
    # wrappers hold no reference, and binding the plan to its declared
    # invocation needs the identity of the tensors it compiled against.
    weight_owner_gate: torch.Tensor | None = None
    weight_owner_down: torch.Tensor | None = None
    # Source-route identity (eighth S3 review): virtual_topk_ids is a
    # DERIVED copy, so its pointer never matches a real routing view —
    # binding must remember the SOURCE tensors' addresses.
    route_topk_ptr: int = 0
    route_slots_ptr: int = 0

    def require_binding(self, site: str, weight: torch.Tensor, routing) -> None:
        """A declared invocation must be THIS plan's fixture AND site.

        Scope (tenth S3 review): identity is enforced at the POINTER level
        (addresses, shapes, strides, domains). IN-PLACE mutation of the
        source routing buffers keeps its addresses and is NOT detected
        here; build_metadata(verify=True) catches it only through group
        counts and the valid-pair set. Full content versioning is
        deliberately out of scope while CuTe is a rejected lab candidate.

        Eighth S3 review closed two fail-open holes in the seventh-review
        check: (a) a weight matching EITHER site's owner passed, so a
        gate/up call could hand the down weight and silently execute the
        captured gate weight — now checked against the SITE's own owner;
        (b) routing was rejected only when pointer AND shape both differed,
        and the pointer could never match (the plan's ids are a copy) — now
        the supplied view's SOURCE tensor addresses must match.
        """
        owner = self.weight_owner_gate if site == "gate_up" else self.weight_owner_down
        if (
            owner is None
            or owner.data_ptr() != weight.data_ptr()
            or owner.shape != weight.shape
            or owner.stride() != weight.stride()
        ):
            raise ValueError(
                f"cutedsl_plan's {site} weight is not the tensor this "
                "invocation declares (address, shape, and strides must all "
                "match — ninth S3 review: same-storage aliases passed) — "
                "plans bind per fixture and per site"
            )
        if routing is None:
            raise ValueError(
                "a declared cutedsl invocation must supply its routing view "
                "— binding cannot be skipped (ninth S3 review)"
            )
        if (
            routing.topk_ids.data_ptr() != self.route_topk_ptr
            or routing.token_slots.data_ptr() != self.route_slots_ptr
            or routing.topk_ids.shape != self.virtual_topk_ids.shape
        ):
            raise ValueError(
                "cutedsl_plan was dispatched from a different route than "
                "the declared routing view (source tensor mismatch)"
            )
        if routing.num_virtual_experts != self.num_groups:
            raise ValueError(
                "the declared routing view's virtual-expert domain "
                f"({routing.num_virtual_experts}) is not this plan's "
                f"dispatch domain ({self.num_groups}) — same source "
                "tensors, different factor semantics (ninth S3 review)"
            )

    # ---- metadata ----

    def build_metadata(self, verify: bool = True) -> None:
        """Re-run dispatch + schedules for the SAME route contents.

        STRICTLY same-route-rebuild (eighth S3 review): m_max was sized for
        the route contents at plan build, and the dispatch kernel has no
        overflow guard — different route contents with a wider group would
        silently spill rows into the next group's storage. ``verify=True``
        (the default) re-checks the group sizes against the sized m_max and
        raises on drift; a timed thunk may pass ``verify=False`` ONLY after
        a verified call on the identical route (dispatch counts are
        deterministic, so one verification covers every replay).
        """
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (
            build_dual_stage_schedules,
        )

        masked_m, src2dst = fused_moe_dispatch_index(
            self.virtual_topk_ids, self.num_groups, self.m_max
        )
        if verify and int(masked_m.max().item()) > self.m_max:
            raise ValueError(
                f"route contents changed: widest group "
                f"{int(masked_m.max().item())} exceeds the sized m_max="
                f"{self.m_max}; rebuild the plan for the new route"
            )
        if verify:
            # Ninth S3 review: an m_max bound alone does not prove the
            # SAME route — a within-capacity content change (e.g. sentinel
            # validity flips) would leave valid_pair_ids stale and index
            # undefined src2dst entries. The plan is fixture-only: group
            # counts and the valid-pair set must match the build exactly.
            fresh_valid = (
                (self.virtual_topk_ids.view(-1) >= 0).nonzero(as_tuple=False).view(-1)
            )
            if not torch.equal(masked_m, self.masked_m) or not torch.equal(
                fresh_valid, self.valid_pair_ids
            ):
                raise ValueError(
                    "route contents changed since plan build (group counts "
                    "or valid-pair set differ) — this plan is fixture-only; "
                    "rebuild it for the new route"
                )
        self.masked_m.copy_(masked_m)
        self.src2dst.copy_(src2dst)
        # Seventh S3 review: the atomics assign a FRESH within-group order
        # every rebuild, so the inverse mapping scatter/down-staging use
        # must be refreshed with it or outputs silently permute.
        self.src2dst_valid.copy_(
            src2dst.index_select(0, self.valid_pair_ids).to(torch.int64)
        )
        schedule_gate, tiles_gate, schedule_down, tiles_down = (
            build_dual_stage_schedules(
                self.masked_m,
                m_max=self.m_max,
                token_width=self.config.token_width,
                n_gemm1=self.n_gate,
                n_gemm2=self.n_down,
                output_width=self.config.output_width,
            )
        )
        self.schedule_gate.copy_(schedule_gate)
        self.tiles_gate.copy_(tiles_gate)
        self.schedule_down.copy_(schedule_down)
        self.tiles_down.copy_(tiles_down)

    # ---- staging ----

    def stage_gate(self, hidden_states: torch.Tensor) -> None:
        fill_gateup_input_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            None,
            self.staged_gate,
            None,
            self.src2dst,
            self.virtual_topk_ids,
            self.top_k,
            self.hidden_size,
            0,
            self.m_max,
            0,
            0,
            BLOCK_SIZE=1024,
            IS_FP8=False,
            SCALE_MN_MAJOR=False,
        )

    def stage_down(self, act_pairs: torch.Tensor) -> None:
        """``act_pairs`` is PAIR-major [P, I] (the down site's input)."""
        self.staged_down.view(-1, self.act_size).index_copy_(
            0,
            self.src2dst_valid,
            act_pairs.index_select(0, self.valid_pair_ids),
        )

    # ---- GEMM + scatter ----

    def _launch(self, site: str) -> None:
        import cuda.bindings.driver as cuda_driver

        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
            as_dynamic_cute_tensor as dyn,
        )

        if site == "gate_up":
            compiled, a, c = self.compiled_gate, self.staged_gate, self.c_gate
            b_arg, schedule, tiles = (
                self.b_arg_gate,
                self.schedule_gate,
                self.tiles_gate,
            )
        else:
            compiled, a, c = self.compiled_down, self.staged_down, self.c_down
            b_arg, schedule, tiles = (
                self.b_arg_down,
                self.schedule_down,
                self.tiles_down,
            )
        stream = cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        compiled(
            dyn(a, leading_dim=2),
            b_arg,
            dyn(c, leading_dim=2),
            dyn(self.masked_m, leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            stream,
        )

    def scatter(self, site: str, bridge_out: torch.Tensor) -> None:
        c = self.c_gate if site == "gate_up" else self.c_down
        bridge_out.index_copy_(
            0,
            self.valid_pair_ids,
            c.view(-1, c.shape[2]).index_select(0, self.src2dst_valid),
        )

    # ---- arm compositions ----

    def run_gate_up(
        self, hidden_states: torch.Tensor, bridge_out: torch.Tensor
    ) -> None:
        """Prepared-boundary thunk: stage + GEMM + scatter (metadata prebuilt)."""
        self.stage_gate(hidden_states)
        self._launch("gate_up")
        self.scatter("gate_up", bridge_out)

    def run_down(self, act_pairs: torch.Tensor, bridge_out: torch.Tensor) -> None:
        self.stage_down(act_pairs)
        self._launch("down")
        self.scatter("down", bridge_out)

    def gemm_only(self, site: str) -> None:
        """Diagnostic ideal bound: what a fused gather-GEMM could not beat."""
        self._launch(site)


def build_cutedsl_lora_a_plan(
    *,
    fused_route: RouteView,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    config: CutedslAConfig,
) -> CutedslLoraAPlan:
    """Size, compile, and pre-run the metadata for one fixture.

    ``fused_route`` must be the ROUTE_FUSED_IDS view of the SAME case route
    the Triton arms' aligned plans were built from — one id domain, one
    grouping, declared by construction.
    """
    virtual_topk_ids = fused_route.virtual_topk_ids.to(torch.int32).contiguous()
    device = virtual_topk_ids.device
    num_groups = fused_route.num_virtual_experts
    top_k = virtual_topk_ids.shape[1]
    n_gate, hidden_size = gate_up_weight.shape[1], gate_up_weight.shape[2]
    n_down, act_size = down_weight.shape[1], down_weight.shape[2]
    # Fail-fast contracts (seventh S3 review) — all BEFORE any compile or
    # staging-buffer allocation:
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (  # noqa: E501
        MAX_EXPERTS,
    )

    if num_groups > MAX_EXPERTS:
        raise ValueError(
            f"{num_groups} virtual groups exceed the direct schedule's "
            f"{MAX_EXPERTS}-group packing"
        )
    if not gate_up_weight.is_contiguous() or not down_weight.is_contiguous():
        raise ValueError(
            "factor weights must be contiguous — a silent .contiguous() "
            "copy here would wrap a temporary whose Torch owner dies"
        )
    if gate_up_weight.shape[0] != num_groups or down_weight.shape[0] != num_groups:
        if gate_up_weight.shape[0] != down_weight.shape[0]:
            raise NotImplementedError(
                "this plan drives BOTH sites off one fused-id dispatch, so "
                "the sites must share a virtual-group count; a shared-outer "
                "gate/up A (V = L_cap) with a per-expert down (V = E x "
                "L_cap) needs two dispatch domains — the shared site runs "
                "through lora_a_shared/SGMV instead (plan section 63.18)"
            )
        raise ValueError(
            "factor weights must be [num_virtual_experts, N, K]; got "
            f"{tuple(gate_up_weight.shape)} / {tuple(down_weight.shape)} for "
            f"V={num_groups}"
        )

    # Sizing pass: masked_m only (m_max=1 makes src2dst meaningless), then
    # right-size m_max to the widest group, token-width aligned.
    masked_m_probe, _ = fused_moe_dispatch_index(virtual_topk_ids, num_groups, 1)
    widest = int(masked_m_probe.max().item())
    m_max = max(
        (widest + config.token_width - 1) // config.token_width * config.token_width,
        config.token_width,
    )

    flat = virtual_topk_ids.view(-1)
    valid_pair_ids = (flat >= 0).nonzero(as_tuple=False).view(-1)
    plan = CutedslLoraAPlan(
        config=config,
        num_groups=num_groups,
        m_max=m_max,
        top_k=top_k,
        hidden_size=hidden_size,
        act_size=act_size,
        n_gate=n_gate,
        n_down=n_down,
        virtual_topk_ids=virtual_topk_ids,
        valid_pair_ids=valid_pair_ids,
        masked_m=torch.empty(num_groups, dtype=torch.int32, device=device),
        src2dst=torch.empty(flat.numel(), dtype=torch.int32, device=device),
        src2dst_valid=torch.empty(
            valid_pair_ids.numel(), dtype=torch.int64, device=device
        ),
        staged_gate=torch.zeros(
            (num_groups, m_max, hidden_size), dtype=torch.bfloat16, device=device
        ),
        staged_down=torch.zeros(
            (num_groups, m_max, act_size), dtype=torch.bfloat16, device=device
        ),
        c_gate=torch.empty(
            (num_groups, m_max, n_gate), dtype=torch.bfloat16, device=device
        ),
        c_down=torch.empty(
            (num_groups, m_max, n_down), dtype=torch.bfloat16, device=device
        ),
        schedule_gate=torch.empty(0, dtype=torch.int32, device=device),
        tiles_gate=torch.empty(1, dtype=torch.int32, device=device),
        schedule_down=torch.empty(0, dtype=torch.int32, device=device),
        tiles_down=torch.empty(1, dtype=torch.int32, device=device),
        compiled_gate=_compiled_masked_gemm(
            device=device, config=config, num_groups=num_groups, n=n_gate, k=hidden_size
        ),
        compiled_down=_compiled_masked_gemm(
            device=device, config=config, num_groups=num_groups, n=n_down, k=act_size
        ),
        b_arg_gate=None,
        b_arg_down=None,
    )
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.api import (
        as_dynamic_cute_tensor,
    )
    from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.schedule_builder import (
        build_dual_stage_schedules,
    )

    plan.b_arg_gate = as_dynamic_cute_tensor(gate_up_weight, leading_dim=2)
    plan.b_arg_down = as_dynamic_cute_tensor(down_weight, leading_dim=2)
    plan.weight_owner_gate = gate_up_weight
    plan.weight_owner_down = down_weight
    plan.route_topk_ptr = fused_route.topk_ids.data_ptr()
    plan.route_slots_ptr = fused_route.token_slots.data_ptr()

    # Real dispatch + schedules once at build; the schedule buffers keep the
    # builder's own capacity sizing, so build_metadata can copy_ in place.
    masked_m, src2dst = fused_moe_dispatch_index(virtual_topk_ids, num_groups, m_max)
    plan.masked_m.copy_(masked_m)
    plan.src2dst.copy_(src2dst)
    plan.src2dst_valid.copy_(src2dst.index_select(0, valid_pair_ids).to(torch.int64))
    schedule_gate, tiles_gate, schedule_down, tiles_down = build_dual_stage_schedules(
        plan.masked_m,
        m_max=m_max,
        token_width=config.token_width,
        n_gemm1=n_gate,
        n_gemm2=n_down,
        output_width=config.output_width,
    )
    plan.schedule_gate = schedule_gate
    plan.tiles_gate = tiles_gate
    plan.schedule_down = schedule_down
    plan.tiles_down = tiles_down
    return plan
