import asyncio
import threading

import uvloop

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from .engine import Engine, _set_envs_and_config
from .runtime import Runtime, launch_server
