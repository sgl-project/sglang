"""Draft-model regions nested inside unified sub-pools' slot entries.

The nesting, outermost first:

  sub-pool  a named pool inside the unified pool, with its own grow direction
            and frontier. HOST is the role it plays when its entries also
            carry draft bytes; only "full" hosts a draft here.
  slot      one token entry of that sub-pool. The draft's KV becomes extra
            parts of every slot, after the host's own parts at
            `draft_offset_in_entry()`, so ONE slot id per token covers target
            and draft: one allocation, one free, one whole-page relocation
            carry both.
  region    the draft geometry inside the host's entries. A region is not a
            `SubPoolSpec`: no grow direction, no frontier logic, only geometry.
  lane      one position in the region's layer dimension -- one draft layer of
            one runner, present in every slot.

Runners are the draft's execution copies, and the two head shapes place them
oppositely. A replicated head gives EVERY runner all the draft's layers, so
each needs its own lanes or the runners clobber each other's KV (3 runners x
2 layers = 6 lanes). A per-depth head (depth 0 predicts the next token,
depth 1 the one after, and so on) gives each runner ONE depth (8 layers
across 2 runners = 2 lanes). Lane count is therefore not layer count, in
either direction. Each runner owns a contiguous run of lanes, and the runs
tile the region in runner order.

Three phases, three moments in boot, and what carries each:

  PLAN   on the TARGET, before any memory exists
    KVCacheConfigurator._fused_draft_decision()
      draft_kv_profile(draft_model_config)  what the checkpoint asks for
      place_fused_draft(profile, runners)   admit, or decline with a reason
    KVCacheConfigurator.fused_entry_bytes() prices it for the boot solve

  CARVE  as the byte buffer is cut
    KVCacheConfigurator._fused_draft_for_pool_factory()  resolve once
      init_unified_*_pools(fused_draft=...)
      MHASubPoolSpec.draft_offset_in_entry() / .entry_bytes()
      UnifiedKVPool.build_dense_draft_views()

  BIND   on each DRAFT runner, built after the target
    KVCacheConfigurator._fused_draft_from_target_buffer(alloc)
      bind_fused_draft(...) -> UnifiedDraftKVPool
    KVIndexTranslator: fused_draft_host_allocator(pool) is alloc -> translate

A draft whose layers need another sub-pool kind declines to a private pool.
"""

from typing import List, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.layout.page_major import DensePart


class DenseDraftRegion(msgspec.Struct, frozen=True, kw_only=True):
    """Geometry of the DRAFT model's K/V rows fused into every slot of a host
    sub-pool. The draft is a separate checkpoint, so its head geometry and
    layer count differ from the host's; its K and V rows are two more parts of
    the host entry, indexed by the host's physical token id."""

    lane_num: int
    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None

    def validate(self) -> None:
        assert self.lane_num > 0, f"lane_num must be positive; got {self.lane_num}"
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        v = self.resolved_v_head_dim()
        assert v > 0, f"v_head_dim must be positive; got {v}"

    def resolved_v_head_dim(self) -> int:
        return self.head_dim if self.v_head_dim is None else self.v_head_dim

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.resolved_v_head_dim() * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        """Draft bytes per slot, before the host entry's alignment."""
        return self.lane_num * (self.k_row_bytes() + self.v_row_bytes())

    def parts(self, offset_bytes: int) -> Tuple[DensePart, DensePart]:
        """The draft's K and V parts, laid out from ``offset_bytes`` inside
        the host entry."""
        layer_stride = self.k_row_bytes() + self.v_row_bytes()
        return (
            DensePart(
                name="draft_k",
                offset_bytes=offset_bytes,
                layer_stride_bytes=layer_stride,
                layer_num=self.lane_num,
                row_shape=(self.head_num, self.head_dim),
                dtype=self.store_dtype,
            ),
            DensePart(
                name="draft_v",
                offset_bytes=offset_bytes + self.k_row_bytes(),
                layer_stride_bytes=layer_stride,
                layer_num=self.lane_num,
                row_shape=(self.head_num, self.resolved_v_head_dim()),
                dtype=self.store_dtype,
            ),
        )


class FusedDraftPlacement(msgspec.Struct, frozen=True, kw_only=True):
    """Where every draft runner's layers live inside the "full" sub-pool: the
    region, and each runner's lane count in runner order. Built once on the
    target, stored on the `UnifiedKVPool`, and read back by each draft
    runner, so the two sides cannot disagree on a lane.
    """

    region: DenseDraftRegion
    runner_lane_counts: Tuple[int, ...]

    def __post_init__(self):
        assert self.runner_lane_counts, "a placement needs at least one draft runner"
        assert sum(self.runner_lane_counts) == self.region.lane_num, (
            f"runner lane counts {self.runner_lane_counts} must tile "
            f"range({self.region.lane_num})"
        )

    def lanes_for(self, runner: int) -> range:
        start = sum(self.runner_lane_counts[:runner])
        return range(start, start + self.runner_lane_counts[runner])


class DraftKVGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """Per-GPU K/V row geometry of one kind of draft attention layer."""

    head_num: int
    head_dim: int
    v_head_dim: int


class DraftKVProfile(msgspec.Struct, frozen=True, kw_only=True):
    """What the draft checkpoint asks of the host sub-pools.

    ``num_depths`` > 1 marks a per-depth head (one transformer block per MTP
    depth, served by one runner each under multi-layer EAGLE); otherwise
    every runner serves all ``num_layers`` layers.
    """

    num_layers: int
    full: DraftKVGeometry
    swa_layer_ids: Tuple[int, ...] = ()
    num_depths: int = 1


def draft_kv_profile(
    draft_model_config, *, num_layers: int, attn_tp_size: int
) -> DraftKVProfile:
    """The profile of a draft `ModelConfig`, heads divided by attn_tp the way
    the target divides its own (drafts never join the DCP group)."""
    mc = draft_model_config
    swa_layer_ids: Tuple[int, ...] = ()
    if mc.is_hybrid_swa and not mc.is_deepseek_v4_arch:
        swa_layer_ids = tuple(int(i) for i in mc.swa_attention_layer_ids)
    num_depths = mc.num_nextn_predict_layers
    return DraftKVProfile(
        num_layers=int(num_layers),
        full=DraftKVGeometry(
            head_num=int(mc.get_num_kv_heads(attn_tp_size)),
            head_dim=int(mc.head_dim),
            v_head_dim=int(mc.v_head_dim),
        ),
        swa_layer_ids=swa_layer_ids,
        num_depths=1 if num_depths is None else int(num_depths),
    )


class FusedDraftDecision(msgspec.Struct, frozen=True, kw_only=True):
    """`place_fused_draft`'s answer: the placement, or why the draft keeps a
    private pool. Neither means fusion simply does not apply."""

    placement: Optional[FusedDraftPlacement] = None
    declined: Optional[str] = None


def _runner_layer_counts(
    profile: DraftKVProfile, num_runners: int
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[str]]:
    """Per runner, its (full, swa) layer counts; or why no runner layout exists."""
    if profile.num_depths <= 1:
        num_swa = len(profile.swa_layer_ids)
        return [(profile.num_layers - num_swa, num_swa)] * num_runners, None
    if num_runners == 1:
        return None, (
            f"a per-depth draft head ({profile.num_depths} depths) needs one "
            "runner per depth (multi-layer EAGLE)"
        )
    if num_runners > profile.num_depths:
        return None, (
            f"{num_runners} draft runners exceed the head's {profile.num_depths} depths"
        )
    swa = set(profile.swa_layer_ids)
    return [(0, 1) if r in swa else (1, 0) for r in range(num_runners)], None


def place_fused_draft(
    *,
    profile: DraftKVProfile,
    num_runners: int,
    store_dtype: torch.dtype,
) -> FusedDraftDecision:
    """Assign every draft layer of every runner to the host sub-pool whose
    lifetime covers what the layer reads: a full-attention layer rides in
    ``"full"``. A layer kind no host arm serves declines the whole draft to
    its private pool."""
    counts, reason = _runner_layer_counts(profile, num_runners)
    if counts is None:
        return FusedDraftDecision(declined=reason)
    num_swa = sum(swa for _, swa in counts)
    if num_swa:
        return FusedDraftDecision(
            declined=(
                f"the draft has {num_swa} SWA layer(s) of its own, which the "
                "fused dense pool cannot serve"
            )
        )
    geometry = profile.full
    if geometry.head_dim != geometry.v_head_dim:
        return FusedDraftDecision(
            declined=(
                "the draft's K/V rows are asymmetric "
                f"(head_dim={geometry.head_dim}, v_head_dim={geometry.v_head_dim}), "
                "which is not admitted yet"
            )
        )
    full_counts = tuple(full for full, _ in counts)
    region = DenseDraftRegion(
        lane_num=sum(full_counts),
        head_num=geometry.head_num,
        head_dim=geometry.head_dim,
        v_head_dim=geometry.v_head_dim,
        store_dtype=store_dtype,
    )
    return FusedDraftDecision(
        placement=FusedDraftPlacement(region=region, runner_lane_counts=full_counts)
    )
