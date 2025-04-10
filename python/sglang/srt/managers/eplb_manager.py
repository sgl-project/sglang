from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage


class EPLBManager:
    def __init__(self):
        self._expert_distribution_storage = ExpertDistributionStorage()

    async def rebalance_experts(self):
        TODO
