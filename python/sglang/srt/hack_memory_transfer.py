import threading


def run_memory_transfer_experiment():
    thread = threading.Thread(target=_thread_entrypoint)
    thread.start()


def _thread_entrypoint():
    TODO
