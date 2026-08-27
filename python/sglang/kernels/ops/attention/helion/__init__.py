"""Helion attention kernels."""

# K3 exposes 12 local value heads at TP=8. Lower value-head counts share the
# same small-head decode regime.
KDA_SMALL_VALUE_HEAD_THRESHOLD = 12
