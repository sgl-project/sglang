# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np


class BaseScheduler(object):

    def schedule(self, n, **kwargs):
        raise NotImplementedError


class LambdaWarmUpCosineFactorScheduler(BaseScheduler):
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, f_min, f_max, f_start, max_decay_steps, verbosity_interval=0, **ignore_kwargs):
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.f_start}")
        if n < self.lr_warm_up_steps:
            f = (self.f_max - self.f_start) / self.lr_warm_up_steps * n + self.f_start
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            f = self.f_min + 0.5 * (self.f_max - self.f_min) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)
