"""Plan-driven dense LoRA with shared routing, scratch and base-GEMM overlap."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
import triton

from sglang.kernels.ops.lora.common.lora_a import (
    grouped_lora_a,
    per_row_lora_a,
    shrink_column_tiles,
)
from sglang.kernels.ops.lora.common.lora_b import (
    grouped_lora_b,
    per_row_lora_b,
    slice_geometry,
)
from sglang.kernels.ops.lora.common.route_view import RouteView, RouteViewKind
from sglang.kernels.ops.lora.common.routing import build_route
from sglang.srt.lora.dense.plan import (
    AFamily,
    BFamily,
    DenseLoraKind,
    DensePlan,
    DensePlanTable,
    Overlap,
    load_plans,
)
from sglang.srt.lora.utils import Phase, architecture_for_capability
from sglang.srt.lora.workspace import LoraWorkspace


class Bridge(NamedTuple):
    """A shrink result and how the expand reads it: ``planes`` > 1 = the
    split-K shrink's fp32 planes [planes, rows, N], summed on load;
    ``slot_planes`` = one [rows, N] plane per adapter slot (all_slots)."""

    tensor: torch.Tensor
    planes: int = 1
    slot_planes: bool = False


class DenseLoraRunner:
    def __init__(
        self,
        workspace: LoraWorkspace,
        *,
        max_loras: int,
        device: torch.device,
        architecture: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.max_loras = max_loras
        self.device = device
        if architecture is None:
            if device.type == "cuda":
                major, _ = torch.cuda.get_device_capability(device)
                architecture = architecture_for_capability(major)
            else:
                architecture = "default"
        self.architecture = architecture
        # Validate the architecture's plan table (and any override) at startup.
        load_plans(architecture)
        self._plan_tables: dict[int, DensePlanTable] = {}
        self.active = False
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.phase = Phase.DECODE
        self.num_tokens = 0
        self._token_slots: torch.Tensor | None = None
        self.token_slots: torch.Tensor | None = None
        self.lora_ranks: torch.Tensor | None = None
        self.scalings: torch.Tensor | None = None
        self.reset_routes()
        self.graph_mode = False

    def reset_routes(self) -> None:
        """Start a forward without dropping bound metadata or workspace storage."""
        self._raw_route: RouteView | None = None
        self.workspace.routes.clear()

    def begin_batch(
        self,
        *,
        token_slots: torch.Tensor,
        lora_ranks: torch.Tensor,
        scalings: torch.Tensor,
        num_tokens: int,
        phase: Phase,
        graph_mode: bool,
        is_prefill_graph: bool = False,
    ) -> None:
        """Bind the token-to-adapter metadata for this batch."""
        self.reset()
        self.active = True
        self.phase = phase
        self.num_tokens = num_tokens
        self._token_slots = token_slots
        self.token_slots = token_slots[:num_tokens]
        self.lora_ranks = lora_ranks
        self.scalings = scalings
        self.graph_mode = bool(graph_mode)
        # Target verify uses prefill plans inside a decode graph.
        self.workspace.begin_forward(
            graph_mode=graph_mode, is_prefill_graph=is_prefill_graph
        )

    def set_num_tokens(self, num_tokens: int) -> None:
        """Use the forward's input shape, including graph padding."""
        if not 0 <= num_tokens <= self._token_slots.numel():
            raise ValueError("dense LoRA input exceeds its token-to-adapter buffer")
        if num_tokens != self.num_tokens:
            self.reset_routes()
            self.num_tokens = num_tokens
            self.token_slots = self._token_slots[:num_tokens]

    def plan_for(
        self,
        kind: DenseLoraKind,
        max_rank: int,
        in_features: int = 0,
        out_features: int = 0,
        *,
        num_tokens: int,
    ) -> DensePlan:
        self.set_num_tokens(num_tokens)
        table = self._plan_tables.get(max_rank)
        if table is None:
            table = DensePlanTable(self.architecture, max_rank)
            self._plan_tables[max_rank] = table
        return table.plan_for(kind, self.phase, num_tokens, in_features, out_features)

    def route(self, block_size: int) -> RouteView:
        """The block route, built once per forward for each block size: tokens
        grouped by adapter slot, capacity independent of the request count."""
        return self._cached_route(("sorted", block_size))

    def _cached_route(self, key: tuple) -> RouteView:
        return self.workspace.route(
            self.token_slots, (*key, self.max_loras), lambda: self._build_route(key)
        )

    def _build_route(self, key: tuple) -> RouteView:
        if key[0] == "pair":
            route = self._build_pair_route(*key[1:])
        else:
            block_size = key[1]
            route = build_route(
                self.token_slots,
                max_loras=self.max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=self.workspace,
                tensor_prefix=f"dense:sorted:{block_size}",
            )
        return route

    def pair_route(
        self, num_heads: int, block_size: int, *, num_tokens: int, aligned: bool = True
    ) -> RouteView:
        """Sort (token, head) pairs by slot*H + head, once per forward."""
        self.set_num_tokens(num_tokens)
        return self._cached_route(("pair", block_size, num_heads, aligned))

    def _build_pair_route(
        self, block_size: int, num_heads: int, aligned: bool
    ) -> RouteView:
        rows = self.num_tokens * num_heads
        device = self.token_slots.device
        head_ids = self.workspace.tensor(
            f"dense:pairs:{num_heads}:heads",
            (rows, 1),
            dtype=torch.int32,
            device=device,
        )
        # Rewritten on replay; workspace scratch does not persist between batches.
        head_ids.copy_(
            torch.arange(num_heads, dtype=torch.int32, device=device)
            .expand(self.num_tokens, num_heads)
            .reshape(rows, 1)
        )
        row_slots = self.workspace.tensor(
            f"dense:pairs:{num_heads}:slots",
            (rows,),
            dtype=torch.int32,
            device=device,
        )
        row_slots.copy_(
            self.token_slots[:, None].expand(self.num_tokens, num_heads).reshape(rows)
        )
        route = build_route(
            row_slots,
            group_ids=head_ids,
            groups_per_slot=num_heads,
            max_loras=self.max_loras,
            block_size=block_size,
            view=RouteViewKind.ALIGNED if aligned else RouteViewKind.RAW,
            workspace=self.workspace,
            tensor_prefix=f"dense:pairs:{num_heads}:{block_size}",
        )
        return route

    def raw_route(self) -> RouteView:
        if self._raw_route is None:
            self._raw_route = RouteView(
                view=RouteViewKind.RAW,
                block_size=1,
                token_slots=self.token_slots,
                group_ids=None,
                groups_per_slot=1,
                max_loras=self.max_loras,
            )
        return self._raw_route

    def run_a(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        *,
        stack: int,
        plan: DensePlan,
        lora_ranks: torch.Tensor | None = None,
        windowed: bool = False,
    ) -> Bridge:
        """The bridge ``[tokens, stack * rank_max]`` with slice ``s`` of a
        rank-``r`` slot at ``[s * r, (s + 1) * r)``, and how the expand reads
        it: the split-K planes shrink returns its fp32 planes, the all_slots
        shrink one plane per slot (see ``Bridge``). ``windowed``: rank block
        ``s`` shrinks input columns ``[s * K, (s + 1) * K)`` (a compact
        per-expert A over the experts' concatenated input); every block runs at
        its full width, the per-slot ranks are not applied."""
        num_tokens = x.shape[0]
        slots, n_max, k = a.shape
        # A caller with max-rank-spaced, zero-filled slots (the MoE-layout
        # pool buffers) passes constant ranks; the batch's ranks otherwise.
        ranks = lora_ranks if lora_ranks is not None else self.lora_ranks
        match plan.a_family:
            case AFamily.GROUPED:
                split_k = int(plan.a_tiles.get("SPLIT_K", 1))
                mode = str(plan.a_tiles.get("SPLIT_MODE", "serial"))
                bridge = self.workspace.tensor(
                    "dense:bridge",
                    (num_tokens, n_max),
                    dtype=x.dtype,
                    device=x.device,
                )
                partial = counters = None
                if split_k > 1:
                    partial = self.workspace.tensor(
                        "dense:bridge_partial",
                        (split_k, num_tokens, n_max),
                        dtype=torch.float32,
                        device=x.device,
                    )
                    if mode == "serial":
                        route = self.route(plan.block_size)
                        tiles = (
                            triton.cdiv(route.sorted_pair_ids.numel(), route.block_size)
                            * shrink_column_tiles(
                                int(plan.a_tiles["BLOCK_SIZE_N"]),
                                n_max,
                                stack,
                                windowed,
                            )[1]
                        )
                        # Zero at rest; the last-arriving program resets its tile.
                        counters = self.workspace.tensor(
                            "dense:splitk_counters",
                            (tiles,),
                            dtype=torch.int32,
                            device=x.device,
                            zero_on_first_allocation=True,
                        )
                grouped_lora_a(
                    x,
                    a,
                    bridge,
                    self.route(plan.block_size),
                    config=plan.a_tiles,
                    pair_input=True,
                    lora_ranks=ranks,
                    stack=stack,
                    partial=partial,
                    counters=counters,
                    windowed=windowed,
                )
                if split_k > 1 and mode == "planes":
                    return Bridge(partial, planes=split_k)
                return Bridge(bridge)
            case AFamily.PER_ROW:
                bridge = self.workspace.tensor(
                    "dense:bridge",
                    (num_tokens, n_max),
                    dtype=x.dtype,
                    device=x.device,
                )
                per_row_lora_a(
                    x,
                    a,
                    bridge,
                    self.raw_route(),
                    config=plan.a_tiles,
                    stack=stack,
                    windowed=windowed,
                )
                return Bridge(bridge)
            case AFamily.ALL_SLOTS:
                if windowed:  # plan_for never pairs them: a forced plan did
                    raise NotImplementedError(
                        "the all-slots GEMM cannot window per rank block"
                    )
                flat = self.workspace.tensor(
                    "dense:bridge_all",
                    (num_tokens, slots * n_max),
                    dtype=x.dtype,
                    device=x.device,
                )
                torch.matmul(x, a.reshape(slots * n_max, k).t(), out=flat)
                return Bridge(
                    flat.view(num_tokens, slots, n_max).transpose(0, 1),
                    slot_planes=True,
                )
            case _:
                raise NotImplementedError(
                    f"no engine executor for A family {plan.a_family.value!r}"
                )

    def run_b(
        self,
        bridge: Bridge,
        b: torch.Tensor,
        out: torch.Tensor,
        *,
        offsets: tuple[int, ...],
        plan: DensePlan,
        add_inplace: bool = True,
        lora_ranks: torch.Tensor | None = None,
        scalings: torch.Tensor | None = None,
        bridge_slices: int = 0,
    ) -> torch.Tensor:
        """``offsets`` (host, [stack+1]) are the slice rows in B and the output
        columns; ``bridge_slices`` > 0 has slice s read
        bridge block s % bridge_slices. ``lora_ranks`` / ``scalings`` override the batch's
        (MoE-layout buffers pass constant ranks and unit scalings)."""
        ranks = lora_ranks if lora_ranks is not None else self.lora_ranks
        scales = scalings if scalings is not None else self.scalings
        b_tiles = plan.b_tiles
        geometry = slice_geometry(offsets, int(b_tiles["BLOCK_SIZE_N"]), b.device)
        match plan.b_family:
            case BFamily.GROUPED:
                grouped_lora_b(
                    bridge.tensor,
                    b,
                    out,
                    self.route(plan.block_size),
                    geometry=geometry,
                    config=b_tiles,
                    add_inplace=add_inplace,
                    zero_sentinel=False,
                    lora_ranks=ranks,
                    scalings=scales,
                    planes=bridge.planes,
                    bridge_slices=bridge_slices,
                )
            case BFamily.PER_ROW:
                per_row_lora_b(
                    bridge.tensor,
                    b,
                    out,
                    self.raw_route(),
                    geometry=geometry,
                    config=b_tiles,
                    add_inplace=add_inplace,
                    zero_sentinel=False,
                    lora_ranks=ranks,
                    scalings=scales,
                    planes=bridge.planes,
                    slot_planes=bridge.slot_planes,
                    bridge_slices=bridge_slices,
                )
            case _:
                raise NotImplementedError(
                    f"no engine executor for B family {plan.b_family.value!r}"
                )
        return out

    def apply(
        self,
        x: torch.Tensor,
        base_fn: Callable[[], torch.Tensor],
        plan: DensePlan,
        *,
        a: torch.Tensor,
        b: torch.Tensor,
        offsets: tuple[int, ...],
        all_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
        lora_ranks: torch.Tensor | None = None,
        scalings: torch.Tensor | None = None,
        bridge_slices: int = 0,
        a_blocks: int | None = None,
        a_windowed: bool = False,
        a_fn: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """base_fn() + LoRA for one site, overlapped as the plan says. ``a`` is
        [slots, a_blocks*rank, K] (``a_blocks`` defaults to the B segments,
        ``len(offsets) - 1``), ``b`` [slots, offsets[-1], rank].
        ``all_reduce`` (row-parallel under TP) reduces base + LoRA once;
        B is replicated across TP ranks.
        ``bridge_slices``: see run_b. ``a_windowed``: rank
        block b of ``a`` shrinks input columns [b*K, (b+1)*K) at its full width
        (see run_a)."""
        self.set_num_tokens(x.shape[0])
        stack = len(offsets) - 1 if a_blocks is None else a_blocks

        def run_a() -> Bridge:
            if a_fn is not None:
                return Bridge(a_fn())  # the embedding shrink: one plain bridge
            return self.run_a(
                x,
                a,
                stack=stack,
                plan=plan,
                lora_ranks=lora_ranks,
                windowed=a_windowed,
            )

        def run_b(bridge: Bridge, out: torch.Tensor, add_inplace: bool) -> None:
            self.run_b(
                bridge,
                b,
                out,
                offsets=offsets,
                plan=plan,
                add_inplace=add_inplace,
                lora_ranks=lora_ranks,
                scalings=scalings,
                bridge_slices=bridge_slices,
            )

        if plan.overlap is Overlap.AB_DELTA:
            # The delta covers the output columns the LoRA slices reach.
            out_end = offsets[-1]
            delta = self.workspace.tensor(
                "dense:delta",
                (x.shape[0], out_end),
                # Embedding inputs are integer token IDs, not activations.
                dtype=b.dtype if a_fn is not None else x.dtype,
                device=x.device,
            )
            delta.zero_()
            base = self.workspace.run_parallel(
                name="dense",
                device=x.device,
                compute=base_fn,
                side=lambda: run_b(run_a(), delta, False),
            )
            base[:, :out_end].add_(delta)
            return all_reduce(base) if all_reduce is not None else base

        if plan.overlap is Overlap.A:
            side_result: list[Bridge] = []
            base = self.workspace.run_parallel(
                name="dense",
                device=x.device,
                compute=base_fn,
                side=lambda: side_result.append(run_a()),
            )
            bridge = side_result[0]
        else:
            base = base_fn()
            bridge = run_a()

        run_b(bridge, base, True)
        return all_reduce(base) if all_reduce is not None else base
