"""Recording adapters and transports shared by AFD pipeline/graph tests.

These objects model stage order and ownership, not tensor arithmetic. Numerical
and storage contracts use real CPU tensors in their respective tests.
"""

from types import SimpleNamespace

from sglang.srt.afd import config as config_mod
from sglang.srt.afd import contracts
from sglang.srt.afd import pipeline as pipeline_mod
from sglang.test.afd.graph_fixtures import make_shape as planned_shape


class FakeTensor:
    dtype = "bfloat16"
    device = "cpu"

    def zero_(self):
        pass

    def copy_(self, source):
        assert source.shape == self.shape

    def __init__(self, rows, hidden=8, stage=None):
        self.shape = (rows, hidden)
        self.stage = stage

    def __getitem__(self, value_slice):
        start = value_slice.start or 0
        stop = value_slice.stop if value_slice.stop is not None else self.shape[0]
        return FakeTensor(stop - start, self.shape[1], self.stage)

    @staticmethod
    def element_size():
        return 2


class EagerGraph:
    retains_backing = False
    capturing = False

    def begin_step(self, **kwargs):
        self.eligible = kwargs["eligible"]
        return contracts.AFDReason.FORWARD_MODE

    def execute_step(self, *, stage_args, compute, **kwargs):
        del kwargs
        self.step_outputs = compute(stage_args)
        return contracts.AFDExecutionResult(
            value=self.step_outputs,
            kind=contracts.AFDExecutionKind.EAGER,
            reason=contracts.AFDReason.FORWARD_MODE,
        )

    @staticmethod
    def end_step():
        return contracts.AFDReason.FORWARD_MODE


class RetainedGraph(EagerGraph):
    """Owns the step and is actually running it as a bucket program.

    retains_backing is what the pipeline reads to learn that, and it is therefore
    also what decides the width every tensor in the region runs at.
    """

    retains_backing = True

    @staticmethod
    def ensure_backing_hbm(*, retained_hbm_bytes):
        del retained_hbm_bytes
        return True


class TraceTransport:
    def __init__(self, descriptor, lane=0, capturing=False, group=1):
        self.descriptor = descriptor
        self.lane = lane
        self.timeline = []
        self.recv_widths = []
        # Attention lanes this rank serves; buffers arrive stage-major, so the
        # stage a buffer belongs to is its index divided by this.
        self.group = group
        # Owning the step is not capturing it: the strategy runs the whole-step
        # region eagerly while it observes. The receive prefetch is gated on this,
        # not on the call site, so the stub has to be able to say either.
        self.capturing = capturing

    def begin_step(self, descriptor):
        assert descriptor is None
        return self.descriptor

    def acquire_buffers(self, *, capacities, hidden_size, dtype, retain, **kwargs):
        del dtype, retain, kwargs
        return tuple(
            FakeTensor(rows, hidden_size, index // self.group)
            for index, rows in enumerate(capacities)
        )

    def receive_dispatch(self, buffers):
        event = None
        for buffer in buffers:
            self.timeline.append(("recv", buffer.stage))
            self.recv_widths.append(buffer.shape[0])
            event = ("ready", buffer.stage)
        return event

    def wait(self, event):
        self.timeline.append(("wait", event[1]))

    def rejoin_streams(self):
        self.timeline.append(("rejoin",))

    def return_result(self, tensors):
        for tensor in tensors:
            self.timeline.append(("return", tensor.stage))

    @staticmethod
    def buffer_hbm_bytes(*, key):
        del key
        return 0

    def release_buffers(self, *, key):
        del key


class FFNAdapter:
    role = contracts.AFDRole.FFN
    hidden_size = 8

    @staticmethod
    def validate_merge_reduce_scatter():
        pass

    def __init__(self, *, layers=1):
        self.num_layers = layers
        self.timeline = None

    def make_ffn_stages(self, *, descriptor, buffers, lane, shape):
        group = shape.group_lanes(ffn_ordinal=lane)
        stages = []
        views = []
        for index in range(len(shape.lane_stage_rows[0])):
            offset = index * len(group)
            lane_views = []
            for member, buffer in zip(group, buffers[offset : offset + len(group)]):
                view = buffer[: descriptor.stage_rows(lane=member)[index]]
                view.stage = index
                lane_views.append(view)
            stages.append(
                SimpleNamespace(
                    index=index,
                    rows=sum(
                        descriptor.stage_rows(lane=member)[index] for member in group
                    ),
                    hidden_states=tuple(lane_views),
                    residual=None,
                    forward_batch=None,
                    merge_sizes=shape.merge_plan(stage=index),
                    merge_lane=lane,
                    group_widths=tuple(
                        shape.lane_bucket_rows[member][index] for member in group
                    ),
                )
            )
            views.append(tuple(lane_views))
        return stages, views

    def local_compute(self, *, layer, stage, hidden_states, **kwargs):
        del kwargs
        self.timeline.append(("local", layer, stage.index))
        return hidden_states, None


class RetainedTransport(TraceTransport):
    def __init__(self, *, role, descriptor=None):
        super().__init__(descriptor)
        self.role = role
        self.registry = {}
        self.retained_allocations = 0
        self.reported_hbm_delta = 0
        self.releases = 0

    def begin_step(self, descriptor):
        if self.role == contracts.AFDRole.ATTENTION:
            assert descriptor is not None
            self.descriptor = descriptor
            return descriptor
        assert descriptor is None
        return self.descriptor

    def acquire_buffers(self, *, key, capacities, hidden_size, dtype, retain):
        del dtype
        if retain and key in self.registry:
            return self.registry[key]
        buffers = tuple(
            FakeTensor(rows, hidden_size, stage)
            for stage, rows in enumerate(capacities)
        )
        if retain:
            self.registry[key] = buffers
            self.retained_allocations += 1
        return buffers

    def buffer_hbm_bytes(self, *, key):
        return self.reported_hbm_delta + sum(
            tensor.shape[0] * tensor.shape[1] * tensor.element_size()
            for tensor in self.registry.get(key, ())
        )

    def release_buffers(self, *, key):
        self.releases += 1
        self.registry.pop(key, None)

    def dispatch(self, tensor):
        self.timeline.append(("dispatch", tensor.stage))

    def receive_return(self, buffer):
        self.timeline.append(("receive_return", buffer.stage))
        self.recv_widths.append(buffer.shape[0])
        return ("ready", buffer.stage)


class DecodeMode:
    @staticmethod
    def is_decode():
        return True

    @staticmethod
    def is_decode_or_idle():
        return True


class DecodeBatch:
    forward_mode = DecodeMode()
    is_extend_in_batch = False


class AttentionAdapter:
    role = contracts.AFDRole.ATTENTION
    num_layers = 2
    hidden_size = 8

    @staticmethod
    def split_step(*, hidden_states, residual, positions, forward_batch, stages):
        del residual, positions, forward_batch
        base, remainder = divmod(hidden_states.shape[0], stages)
        result = []
        for stage in range(stages):
            rows = base + (stage < remainder)
            result.append(
                SimpleNamespace(
                    index=stage,
                    rows=rows,
                    hidden_states=FakeTensor(rows, 8, stage),
                    residual=None,
                    positions=FakeTensor(rows, 1, stage),
                    forward_batch=DecodeBatch(),
                )
            )
        return result

    @staticmethod
    def lane_stage_rows(*, forward_batch, local_rows, lanes, lane):
        del forward_batch, lane
        return (local_rows,) * lanes

    @staticmethod
    def local_compute(*, stage, hidden_states, residual, **kwargs):
        del kwargs
        hidden_states.stage = stage.index
        return hidden_states, residual

    @staticmethod
    def finish_layer(*, stage, ffn_output, residual, **kwargs):
        del kwargs
        ffn_output.stage = stage.index
        return ffn_output, residual

    @staticmethod
    def join_step(*, stages):
        return FakeTensor(sum(stage.rows for stage in stages)), None

    @staticmethod
    def prepare_stage(*, stage):
        del stage

    @staticmethod
    def metadata_guard(*, stage, bucket_rows):
        del stage, bucket_rows
        return None


class PipelineTransport(RetainedTransport):
    def __init__(self, *, role, descriptor=None, group=1, capturing=False):
        super().__init__(role=role, descriptor=descriptor)
        self.group = group
        self.capturing = capturing
        self.packets = []
        self.sent = []
        self.begin_shapes = []

    def acquire_buffers(self, **kwargs):
        buffers = super().acquire_buffers(**kwargs)
        for index, buffer in enumerate(buffers):
            buffer.stage = index // self.group
        return buffers

    def receive_dispatch(self, buffers):
        self.packets.append(tuple(buffers))
        return super().receive_dispatch(buffers)

    def dispatch(self, tensor, **routes):
        self.sent.append((tensor, routes))
        super().dispatch(tensor)


def hidden_ffn(
    matrix,
    layers,
    *,
    graph=None,
    capturing=False,
    ffn_lanes=1,
    extend=False,
    eligible=True,
    lane=0,
):
    adapter = FFNAdapter(layers=layers)
    descriptor = contracts.AFDStepDescriptor(
        kind="STEP",
        step_id=0,
        lane_stage_rows=matrix,
        hidden_size=8,
        dtype="bfloat16",
        num_layers=adapter.num_layers,
        graph_eligible=eligible,
        is_extend_in_batch=extend,
    )
    transport = PipelineTransport(
        role=contracts.AFDRole.FFN,
        descriptor=descriptor,
        group=len(matrix) // ffn_lanes,
        capturing=capturing,
    )
    transport.lane = lane
    adapter.timeline = transport.timeline
    cfg = config_mod.AFDConfig(
        stages=len(matrix[0]),
        lanes=ffn_lanes,
        attention_lanes=len(matrix),
    )
    pipeline = pipeline_mod.AFDFFNPipeline(
        adapter=adapter,
        connector=SimpleNamespace(
            transport=transport,
            graph_strategy=graph or EagerGraph(),
        ),
        config=cfg,
        device="cpu",
        dtype="bfloat16",
        shape_factory=planned_shape,
    )
    return pipeline, adapter, transport


class MatrixAttentionAdapter(AttentionAdapter):
    def __init__(self, matrix, layers):
        self.matrix = matrix
        self.num_layers = layers
        self.calls = []
        self.finished = []

    def split_step(self, **kwargs):
        stages = super().split_step(**kwargs)
        for stage, rows in zip(stages, self.matrix[0]):
            stage.rows = rows
            stage.hidden_states = FakeTensor(rows, 8, stage.index)
            stage.positions = FakeTensor(rows, 1, stage.index)
            if kwargs["residual"] is not None:
                stage.residual = FakeTensor(rows, 8, stage.index)
        return stages

    def lane_stage_rows(self, **kwargs):
        return self.matrix

    def local_compute(self, *, layer, stage, hidden_states, residual, **kwargs):
        self.calls.append((layer, stage.index))
        return hidden_states, residual

    def finish_layer(self, *, layer, stage, ffn_output, residual):
        assert stage.rows, "an empty lane must not run layer postprocessing"
        self.finished.append((layer, stage.index))
        return ffn_output, residual


def hidden_attention(matrix, layers, *, graph=None, residual=False):
    adapter = MatrixAttentionAdapter(matrix, layers)
    transport = PipelineTransport(role=contracts.AFDRole.ATTENTION)
    cfg = config_mod.AFDConfig(
        stages=len(matrix[0]),
        lanes=len(matrix),
    )
    pipeline = pipeline_mod.AFDAttentionPipeline(
        adapter=adapter,
        connector=SimpleNamespace(
            transport=transport,
            graph_strategy=graph or EagerGraph(),
            shutdown_requested=False,
        ),
        config=cfg,
        shape_factory=planned_shape,
    )

    def run(batch=None):
        rows = sum(adapter.matrix[0])
        return pipeline.execute(
            hidden_states=FakeTensor(rows),
            positions=FakeTensor(rows, 1),
            residual=FakeTensor(rows) if residual else None,
            forward_batch=batch or DecodeBatch(),
        )

    return pipeline, adapter, transport, run
