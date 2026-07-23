import logging
import queue
import threading

import torch
from torch.utils.data import _utils

logger = logging.getLogger(__name__)


class _MultiProcessingBase:
    def __init__(self, num_workers, fn, input_queue, output_queue):
        self._num_workers = num_workers
        multiprocessing_context = torch.multiprocessing
        self._multiprocessing_context = multiprocessing_context

        self._worker_result_queue = multiprocessing_context.Queue()
        self._workers_done_event = multiprocessing_context.Event()
        self._num_workers = num_workers
        self._index_queues = []
        self._workers = []
        self._pin_memory = False

        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            if output_queue is None:
                w = multiprocessing_context.Process(target=fn, args=(input_queue,))
            else:
                w = multiprocessing_context.Process(
                    target=fn, args=(input_queue, output_queue)
                )
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            from pickle import PicklingError

            try:
                w.start()
            except (TypeError, AttributeError, PicklingError):
                logger.warning(
                    "Got pickle error when attempting to start a worker Process. "
                    "This might be because the worker Process arguments are not picklable. "
                    "Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms "
                    "to 'forkserver', which requires the worker Process arguments to be picklable. "
                    "You can also try multiprocessing.set_start_method('fork').",
                    stacklevel=2,
                )
                raise
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            current_device_id = torch.accelerator.current_device_index()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    current_device_id,
                    self._pin_memory_thread_done_event,
                    self._pin_memory_device,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue  # type: ignore[assignment]

    @property
    def multiprocessing_context(self):
        return self._multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context) -> None:
        if multiprocessing_context is not None:
            if self._num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = torch.multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            "multiprocessing_context option "
                            f"should specify a valid start method in {valid_start_methods!r}, but got "
                            f"multiprocessing_context={multiprocessing_context!r}"
                        )
                    multiprocessing_context = torch.multiprocessing.get_context(
                        multiprocessing_context
                    )
            else:
                raise ValueError(
                    "multiprocessing_context can only be used with "
                    "multi-process loading (num_workers > 0), but got "
                    f"num_workers={self._num_workers}"
                )

        self._multiprocessing_context = multiprocessing_context

    def _shutdown_workers(self) -> None:
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        if (
            _utils is None
            or _utils.python_exit_status is True
            or _utils.python_exit_status is None
        ):
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, "_pin_memory_thread"):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    # We should be able to join here, but in case anything went
                    # wrong, we set a timeout and if the workers fail to join,
                    # they are killed in the `finally` block.
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False) -> None:
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        if (
            not self._workers_status[worker_id]
            and not self._persistent_workers
            and not shutdown
        ):
            raise AssertionError(
                "Worker status inconsistent when marking worker as unavailable"
            )

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joining is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        if self._workers_done_event.is_set() != shutdown:
            raise AssertionError(
                "_workers_done_event state does not match shutdown flag"
            )

    @staticmethod
    def _clean_up_worker(w) -> None:
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self) -> None:
        self._shutdown_workers()


class _MultiImageProcessingLoader(_MultiProcessingBase):
    def __init__(self, num_workers, fn, input_queue, output_queue):
        super().__init__(num_workers, fn, input_queue, output_queue)


class _MultiImagePreProcess(_MultiProcessingBase):
    def __init__(self, num_workers, fn, input_queue, output_queue):
        super().__init__(num_workers, fn, input_queue, output_queue)
