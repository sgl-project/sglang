"""Throughput metrics for the scheduler to detokenizer routing stage."""

from prometheus_client import Counter


class DetokenizerRouterMetrics:
    def __init__(self):
        self.messages_received = Counter(
            "sglang:detokenizer_router_messages_received_total",
            "Messages received from the scheduler, including control and idle messages",
        )
        self.request_outputs_received = Counter(
            "sglang:detokenizer_router_request_outputs_received_total",
            "Request outputs received from the scheduler, including repeated streaming outputs",
        )
        self.messages_sent = Counter(
            "sglang:detokenizer_router_messages_sent_total",
            "Messages sent to detokenizer workers after grouping or broadcasting",
        )
