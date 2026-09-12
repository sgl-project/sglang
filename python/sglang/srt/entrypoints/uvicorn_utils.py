import signal

import uvicorn
from uvicorn.supervisors import multiprocess


def run_uvicorn_with_sigquit_handler(*args, **kwargs) -> None:
    """Keep SGLang's crash handler active in the multi-tokenizer parent."""
    original_signals = multiprocess.SIGNALS
    try:
        # Uvicorn otherwise captures SIGQUIT without invoking SGLang's cleanup.
        # Exclude it before the supervisor registers its signal handlers.
        multiprocess.SIGNALS = {
            sig: name for sig, name in original_signals.items() if sig != signal.SIGQUIT
        }
        uvicorn.run(*args, **kwargs)
    finally:
        multiprocess.SIGNALS = original_signals
