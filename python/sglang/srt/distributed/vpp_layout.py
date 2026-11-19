from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VPPStage:
    """A logical stage and its contiguous layer range."""

    stage_id: int
    physical_rank: int
    start_layer: int
    end_layer: int

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.start_layer, self.end_layer))


@dataclass(frozen=True)
class VPPLayout:
    """Immutable interleaved pipeline layout.

    Each physical rank owns `virtual_stages` logical stages.
    `partition` specifies the layer count of each logical stage.
    """

    num_hidden_layers: int
    physical_size: int
    virtual_stages: int = 2
    partition: tuple[int, ...] | None = None
    stages: tuple[VPPStage, ...] = field(init=False)

    @property
    def logical_size(self) -> int:
        return self.physical_size * self.virtual_stages

    def __post_init__(self) -> None:
        for name, value in (
            ("num_hidden_layers", self.num_hidden_layers),
            ("physical_size", self.physical_size),
            ("virtual_stages", self.virtual_stages),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        logical_size = self.logical_size
        if self.num_hidden_layers < logical_size:
            raise ValueError(
                f"num_hidden_layers ({self.num_hidden_layers}) must be "
                f"at least the logical stage count ({logical_size})"
            )

        if self.partition is None:
            # Match the existing PP rule: assign extra layers to later stages.
            base, remainder = divmod(self.num_hidden_layers, logical_size)
            partition = tuple(
                base + int(stage_id >= logical_size - remainder)
                for stage_id in range(logical_size)
            )
        else:
            # Normalize to a tuple so callers cannot mutate the partition.
            partition = tuple(self.partition)

        if len(partition) != logical_size:
            raise ValueError(
                f"partition must contain {logical_size} stage layer counts"
            )
        if any(type(count) is not int or count <= 0 for count in partition):
            raise ValueError("Stage layer counts must be positive integers")
        if sum(partition) != self.num_hidden_layers:
            raise ValueError("Stage layer counts must sum to num_hidden_layers")

        stages = []
        start_layer = 0
        for stage_id, count in enumerate(partition):
            end_layer = start_layer + count
            stages.append(
                VPPStage(
                    stage_id=stage_id,
                    physical_rank=stage_id % self.physical_size,
                    start_layer=start_layer,
                    end_layer=end_layer,
                )
            )
            start_layer = end_layer

        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "stages", tuple(stages))

    def stage(self, stage_id: int) -> VPPStage:
        if type(stage_id) is not int or not 0 <= stage_id < self.logical_size:
            raise ValueError(
                f"stage_id must be an integer in [0, {self.logical_size})"
            )
        return self.stages[stage_id]

    def stages_for_rank(self, physical_rank: int) -> tuple[VPPStage, ...]:
        if (
            type(physical_rank) is not int
            or not 0 <= physical_rank < self.physical_size
        ):
            raise ValueError(
                f"physical_rank must be an integer in [0, {self.physical_size})"
            )
        return tuple(
            stage
            for stage in self.stages
            if stage.physical_rank == physical_rank
        )

    def layer_ids_for_rank(self, physical_rank: int) -> tuple[int, ...]:
        return tuple(
            layer_id
            for stage in self.stages_for_rank(physical_rank)
            for layer_id in stage.layer_ids
        )
