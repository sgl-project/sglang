"""
Backend configurations, may vary with different serving platforms.
"""

from dataclasses import dataclass


@dataclass
class BackendConfig:
    extend_dependency_time: float = 0.03


GLOBAL_BACKEND_CONFIG = BackendConfig()
