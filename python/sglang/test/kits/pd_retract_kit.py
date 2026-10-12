import requests

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import assert_process_healthy

_METRICS_REQUEST_TIMEOUT = 30


class PDRetractKit(GSM8KMixin):
    """Forced retraction on a PD decode worker must not take the worker down or
    cost accuracy.

    Mix in ahead of ``PDDisaggregationServerBase``. The decode worker retracts a
    request every ``retract_interval`` forward passes; the GSM8K run goes through
    the router, so every retracted request is recovered by the decode's
    retraction backend and still has to answer correctly.
    """

    retract_interval: int = 20

    @classmethod
    def start_decode(cls):
        cls.extra_decode_env = {
            **cls.extra_decode_env,
            "SGLANG_TEST_RETRACT": "1",
            "SGLANG_TEST_RETRACT_INTERVAL": str(cls.retract_interval),
        }
        # The retraction counter is only exported with metrics on.
        cls.extra_decode_args = [*cls.extra_decode_args, "--enable-metrics"]
        super().start_decode()

    def _decode_retracted_total(self) -> float:
        response = requests.get(
            self.decode_url + "/metrics", timeout=_METRICS_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in response.text.splitlines()
            if line.startswith("sglang:num_retracted_requests_total")
        )

    def test_gsm8k(self):
        super().test_gsm8k()
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
        self.assertGreater(
            self._decode_retracted_total(),
            0,
            "the decode worker never retracted a request; the run proves nothing",
        )
