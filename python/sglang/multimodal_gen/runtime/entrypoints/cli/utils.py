# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import argparse


class RaiseNotImplementedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        raise NotImplementedError(f"The {option_string} option is not yet implemented")
