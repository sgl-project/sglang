from __future__ import annotations


class SchedulerDisaggregationInitMixin:
    """Install the backend-agnostic disaggregation initializer on Scheduler.

    Scheduler keeps the main-branch method body for review friendliness. This
    mixin swaps in the disaggregation implementation after the concrete
    Scheduler class is created, so hidden-state initialization stays owned by
    the disaggregation package.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        if issubclass(cls, SchedulerDisaggregationPrefillMixin):
            cls.init_disaggregation = (
                SchedulerDisaggregationPrefillMixin.init_disaggregation
            )
