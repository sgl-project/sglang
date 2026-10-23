"""Resolved shared-expert layout for one loaded model."""

import msgspec


class SharedExpertsConfig(msgspec.Struct, frozen=True):
    """Immutable layout shared by a model, its routers and its captured graphs."""

    fusion_disabled: bool = False
    num_fused_shared_experts: int = 0
    use_waterfill: bool = False

    @property
    def num_shared_topk_slots(self) -> int:
        """Shared id/weight slots emitted before Waterfill expansion."""
        return 0 if self.use_waterfill else self.num_fused_shared_experts


def default_shared_experts_config(args) -> SharedExpertsConfig:
    """Publication has no loaded model yet; retain only the startup intent."""
    return SharedExpertsConfig(fusion_disabled=args.disable_shared_experts_fusion)
