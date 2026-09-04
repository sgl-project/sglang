from __future__ import annotations

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo


class RacerSpecAlgo(CustomSpecAlgo):
    """Reuse the NGRAM irregular-tree runtime contract for RACER."""

    def is_ngram(self) -> bool:
        return True

    def has_draft_kv(self) -> bool:
        return False

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
