"""A model-free encoder for exercising the real staged gRPC and request lifecycle."""

from sglang.srt.disaggregation.encoder.grpc_server import SGLangEncoderServer
from sglang.srt.disaggregation.encoder.server import MMEncoder


class StagedEncoderFixture(SGLangEncoderServer):
    """Subclasses supply media encoding and the native delivery backend's send/release."""

    def __init__(self, info, *, backend, stage_timeout, ttl, capacity, signer=None):
        self.encoder = MMEncoder.__new__(MMEncoder)
        self.encoder.req_states = {}
        self.encoder.mm_global_cache_enabled = False
        self.encoder.transfer_backend = backend
        self.encoder.delivery = self
        self._configure_staging(
            info, stage_timeout=stage_timeout, ttl=ttl, capacity=capacity, signer=signer
        )
