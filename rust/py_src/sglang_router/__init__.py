# a lightweihgt wrapper on router with argument type and comments
from sglang_router_rs import PolicyType

# no wrapper on policy type => direct export
from .router import Router

__all__ = ["Router", "PolicyType"]

from sglang_router.version import __version__

__all__ += ["__version__"]
