from concurrent import futures

import torch
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch


class LoRAPrefetcher:
    def __init__(self, tp_worker, device):
        self.tp_worker = tp_worker
        self.device = device

        self.load_stream: CudaStream = torch.get_device_module(device).Stream()
        self.load_stream_ctx: CudaStreamContext = torch.get_device_module(
            device
        ).stream(self.load_stream)
        self.lora_prefetch_executor = futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lora_prefetch"
        )
        self.lora_to_prefetch_future = {}

    def prepare_for_prefill(self, running_batch: ScheduleBatch):
        self.loaded_loras = {
            req.lora_id for req in running_batch.reqs if req is not None
        }

    def _do_lora_prefetch_in_thread(self, model_worker_batch: ModelWorkerBatch):
        with self.load_stream_ctx:
            self.tp_worker.fetch_lora_batch(model_worker_batch)
            load_done = torch.get_device_module(self.device).Event()
            load_done.record(self.load_stream)

        return load_done

    def check_loaded_and_maybe_prefetch(
        self,
        req: Req,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache,
        model_config,
        enable_overlap,
        spec_algorithm,
    ):
        if req.lora_id is None or req.lora_id in self.loaded_loras:
            return True

        if req.lora_id not in self.lora_to_prefetch_future:
            new_lora_set = (
                self.loaded_loras | self.lora_to_prefetch_future.keys() | {req.lora_id}
            )
            if self.tp_worker.can_run_lora_batch(new_lora_set):
                # TODO (glenliu21): instead of going through tp_worker, this module should directly interact with LoRAManager
                fetch_lora_batch = ScheduleBatch.init_new(
                    [req],
                    req_to_token_pool,
                    token_to_kv_pool_allocator,
                    tree_cache,
                    model_config,
                    enable_overlap,
                    spec_algorithm,
                )
                fetch_lora_batch.prepare_for_lora_prefetch()
                fetch_model_worker_batch = fetch_lora_batch.get_model_worker_batch()

                future = self.lora_prefetch_executor.submit(
                    self._do_lora_prefetch_in_thread,
                    fetch_model_worker_batch,
                )
                self.lora_to_prefetch_future[req.lora_id] = future

            return False

        future = self.lora_to_prefetch_future[req.lora_id]
        if not future.done():
            return False

        load_done = future.result()
        if not load_done.query():
            return False

        torch.cuda.current_stream().wait_event(load_done)

        del self.lora_to_prefetch_future[req.lora_id]
        self.loaded_loras.add(req.lora_id)

        return True
