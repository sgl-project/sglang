from pathlib import Path


class ExpertDistributionStorage:
    def __init__(self, dir_data):
        self._dir_data = Path(dir_data)

    def save_current(self):
        TODO

    def get_last_snapshot(self):
        return TODO
