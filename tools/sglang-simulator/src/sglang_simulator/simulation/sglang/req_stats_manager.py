from sglang_simulator.simulation.types import RequestStats


class RequestStatsManager:
    """Shared request statistics manager for `Scheduler` and `HicacheController`."""

    def __init__(self):
        self.stats: dict[str, RequestStats] = {}

    def get_req_stats(self, rid: str) -> RequestStats:
        if rid not in self.stats:
            self.stats[rid] = RequestStats(rid=rid)
        return self.stats[rid]

    def get_all_req_stats(self) -> list[RequestStats]:
        return list(self.stats.values())

    def reset(self):
        self.stats.clear()


request_stats_manager = RequestStatsManager()
