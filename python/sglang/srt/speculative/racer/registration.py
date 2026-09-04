from __future__ import annotations

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo


class RacerSpecAlgo(CustomSpecAlgo):
    """Reuse the NGRAM irregular-tree runtime contract for RACER."""

    def is_ngram(self) -> bool:
        return True

    def has_draft_kv(self) -> bool:
        return False

    def carries_draft_hidden_states(self) -> bool:
        # RACER has no draft model and therefore no draft hidden states to
        # transfer across disaggregated prefill/decode boundaries.
        return False

    def need_topk(self) -> bool:
        # RACER computes copy-logit top-k from target verify logits inside its
        # worker; it does not need the generic EAGLE/Standalone top-k plumbing.
        return False

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        # Keep parity with SpeculativeAlgorithm's duck-typed interface even
        # though the first RACER integration runs with overlap disabled.
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device,
            self,
            req_to_token_pool,
            needs_cpu_seq_lens,
            needs_confidence_relay,
        )

    def handle_server_args(self, server_args) -> None:
        from sglang.srt.arg_groups.speculative_hook import _handle_ngram

        if server_args.speculative_ngram_external_corpus_path is not None:
            raise ValueError(
                "RACER does not support --speculative-ngram-external-corpus-path."
            )
        _handle_ngram(server_args)


@SpeculativeAlgorithm.register(
    "RACER",
    supports_overlap=False,
    spec_class=RacerSpecAlgo,
)
def _racer_worker_factory(server_args):
    del server_args
    from sglang.srt.speculative.racer.worker import RACERWorker

    return RACERWorker
