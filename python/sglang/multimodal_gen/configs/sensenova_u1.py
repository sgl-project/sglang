# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the native SenseNova-U1 integration."""

SENSENOVA_U1_REQUEST_EXTRA_KEY = "sensenova_u1"

SENSENOVA_U1_CFG_NORM_CHOICES = (
    "none",
    "global",
    "channel",
    "cfg_zero_star",
)
SENSENOVA_U1_RESOLUTION_ALIGNMENT = 32

DEFAULT_CFG_NORM = "none"
DEFAULT_TIMESTEP_SHIFT = 3.0
DEFAULT_ENABLE_TIMESTEP_SHIFT = True
DEFAULT_CFG_INTERVAL = (0.0, 1.0)
DEFAULT_T_EPS = 0.02
DEFAULT_THINK_MODE = False
