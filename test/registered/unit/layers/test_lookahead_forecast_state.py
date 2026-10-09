import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.lookahead import (
    ForecastState,
    get_forecast_state,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestForecastState(CustomTestCase):
    def test_first_layer_falls_back_to_current_query_and_next_layer_gets_forecast(
        self,
    ):
        forward_batch = SimpleNamespace(model_specific_states=None)
        state = get_forecast_state(forward_batch)

        self.assertIsInstance(state, ForecastState)
        self.assertIsNone(state.for_layer(0))

        forecast = torch.randn(3, 2, 4)
        state.publish(0, forecast)

        self.assertIs(state.for_layer(1), forecast)
        self.assertIsNone(state.for_layer(0))
        self.assertIsNone(state.for_layer(2))

    def test_forecast_state_isolated_between_forward_batches(self):
        first_batch = SimpleNamespace(model_specific_states=None)
        second_batch = SimpleNamespace(model_specific_states=None)

        first_state = get_forecast_state(first_batch)
        second_state = get_forecast_state(second_batch)
        first_forecast = torch.randn(2, 1, 8)
        second_forecast = torch.randn(4, 1, 8)

        first_state.publish(2, first_forecast)
        second_state.publish(2, second_forecast)

        self.assertIsNot(first_state, second_state)
        self.assertIs(get_forecast_state(first_batch).for_layer(3), first_forecast)
        self.assertIs(get_forecast_state(second_batch).for_layer(3), second_forecast)
        self.assertIsNone(get_forecast_state(first_batch).for_layer(4))

    def test_reset_discards_forecast_from_previous_forward(self):
        forward_batch = SimpleNamespace(model_specific_states=None)
        state = get_forecast_state(forward_batch)
        state.publish(0, torch.randn(1, 1, 4))

        state.reset()

        self.assertIsNone(state.for_layer(1))
        self.assertIsNone(state.query)
        self.assertIsNone(state.producer_layer)


if __name__ == "__main__":
    unittest.main()
