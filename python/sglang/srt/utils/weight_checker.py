class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner

    def handle(self, action: str):
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_param":
            self._reset_param()
        elif action == "compare":
            self._compare()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        TODO

    def _reset_param(self):
        TODO

    def _compare(self):
        TODO
