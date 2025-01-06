import threading

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)
