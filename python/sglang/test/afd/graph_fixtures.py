"""Recording graph boundary shared by CPU lifecycle and cache tests."""

from functools import partial

from sglang.srt.afd.cache import make_shape as _make_shape

CAPTURE_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)
make_shape = partial(_make_shape, capture_sizes=CAPTURE_SIZES)


class RecordingProgram:
    def __init__(self, spec, owner):
        self.spec = spec
        self.owner = owner
        self.closed = 0
        self.armed = 0
        self.replays = 0
        self.retained_hbm_bytes = 0

    def replay(self, *, stage_args, stage_rows, forward_batches):
        del stage_rows, forward_batches
        self.replays += 1
        self.owner.replay_ids.append(id(self))
        return self.spec.compute(stage_args)

    def arm_sentinel(self):
        self.armed += 1

    def require_sentinel_written(self):
        pass

    def close(self):
        self.closed += 1


class RecordingDriver:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.programs = []
        self.replay_ids = []
        self.captures = 0
        self.closed = 0

    def capture(self, *, spec):
        self.captures += 1
        if self.fail_at is not None and len(self.programs) == self.fail_at:
            raise RuntimeError("capture failed")
        program = RecordingProgram(spec, self)
        self.programs.append(program)
        return program

    @staticmethod
    def memory_usage(*, device):
        del device
        return (0, 0)

    def close(self):
        self.closed += 1


def role_service(*, driver=None, role=None, device="cuda:0", num_layers=2, **values):
    from sglang.srt.afd.config import AFDConfig
    from sglang.srt.afd.contracts import AFDRole
    from sglang.srt.afd.role_graph import AFDRoleGraphService

    config = AFDConfig(**values)
    service = AFDRoleGraphService(
        capture_sizes=CAPTURE_SIZES,
        role=role or AFDRole.ATTENTION,
        config=config,
        num_layers=num_layers,
        device=device,
        driver=driver or RecordingDriver(),
    )
    return config, service


def role_step(service, config, *, step_id, rows=(3, 3), hidden=16, compute=None):
    import torch

    shape = make_shape(
        lane=0,
        lane_rows=(rows,),
        hidden_size=hidden,
        dtype="bfloat16",
        config=config,
    )
    service.begin_step(
        step_id=step_id,
        shape=shape,
        eligible=True,
        capture=not any(b.shape.digest == shape.digest for b in service._cache.buckets),
    )
    try:
        result = service.execute_step(
            stage_args=tuple(
                (torch.full((r, hidden), float(i + 1)),) for i, r in enumerate(rows)
            ),
            stage_rows=rows,
            forward_batches=(None,) * len(rows),
            compute=compute
            or (lambda staged: tuple((values[0] * 2, None) for values in staged)),
            metadata_guards=(None,) * len(rows),
        )
    finally:
        end = service.end_step()
    return result, end
