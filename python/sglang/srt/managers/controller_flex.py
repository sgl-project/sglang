"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
A controller that manages multiple data parallel workers.
Each data parallel worker can manage multiple tensor parallel workers.
"""
import dataclasses
import logging
import multiprocessing
import multiprocessing.shared_memory
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum, auto

import numpy as np
import torch
import zmq


def _key_match(key0, key1):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def get_match_len(node, key, match_length: int) -> int:
    if len(key) == 0:
        return match_length

    if key[0] in node.children.keys():
        child = node.children[key[0]]
        prefix_len = _key_match(child.key, key)
        match_length += prefix_len
        if prefix_len < len(child.key):
            return match_length
        else:
            return get_match_len(child, key[prefix_len:], match_length)
    else:
        return match_length


import threading
import time

from sglang.srt.managers.controller_single import (
    start_controller_process as start_controller_process_single,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    ControllerInfo,
    FlushCacheReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_parent_process
from sglang.utils import get_cache_info, get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    RESOURCES_AWARE = auto()
    POWER_OF_2_CHOICE = auto()
    PRE_RADIX = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


@dataclasses.dataclass
class WorkerHandle:
    """Store the handle of a data parallel worker."""

    proc: multiprocessing.Process
    queue: multiprocessing.Queue


# class FlexScheduler:
#     """A scheduler which dispatch """


class ControllerMultiFlex:
    """A controller that manages multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args,
    ):
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.model_overide_args = model_overide_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init communication
        context = zmq.Context()
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.controller_port}")

        self.recv_from_tree_cache = context.socket(zmq.PULL)
        self.recv_from_tree_cache.setsockopt(zmq.RCVHWM, 1000)
        self.recv_from_tree_cache.bind(f"tcp://127.0.0.1:41935")

        self.pre_radix = server_args.load_balance_method == "pre_radix"
        self.dp_size = server_args.dp_size

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
            LoadBalanceMethod.RESOURCES_AWARE: self.resources_aware_scheduler,
            LoadBalanceMethod.POWER_OF_2_CHOICE: self.power_of_2_choice,
            LoadBalanceMethod.PRE_RADIX: self.pre_radix_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        self.newest_tree_cache = {}

        # Start data parallel workers
        self.workers = []
        self.controller_info = ControllerInfo(server_args, model_overide_args)
        for i in range(server_args.dp_size):
            self.start_dp_worker(i)

        self.scheduler_time = 0

        self.cnt = 0

        if self.pre_radix:
            self.recv_tree_cache_lock = threading.Lock()
            threading.Thread(target=self.loop_for_recv_tree_cache).start()

    def start_dp_worker(self, dp_worker_id: int):
        tp_size = self.server_args.tp_size

        pipe_controller_reader, pipe_controller_writer = multiprocessing.Pipe(
            duplex=False
        )

        gpu_ids = list(range(dp_worker_id * tp_size, (dp_worker_id + 1) * tp_size))
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=start_controller_process_single,
            args=(
                self.server_args,
                self.port_args,
                pipe_controller_writer,
                self.model_overide_args,
                True,
                gpu_ids,
                dp_worker_id,
                queue,
                self.controller_info,
            ),
        )
        proc.start()

        controller_init_state = pipe_controller_reader.recv()
        if controller_init_state != "init ok":
            raise RuntimeError(
                f"Initialization failed. controller_init_state: {controller_init_state}"
            )
        self.workers.append(
            WorkerHandle(
                proc=proc,
                queue=queue,
            )
        )

    def compute_prefix_length(self, gpu_id, radix_cache, input_ids):
        return gpu_id, get_match_len(radix_cache.root_node, input_ids, 0)

    def pre_radix_scheduler(self, input_requests):
        if len(input_requests) == 0:
            return

        # available_mem = [k.value for k in self.controller_info.available_kv_cache]
        # num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]
        # num_reqs_running = [k.value for k in self.controller_info.running_reqs]

        # all_waitting = False
        # if min(num_reqs_waiting) > 0:
        #     # 最小值都大于0，全部waiting
        #     all_waitting = True
        # else:
        #     # 最小值都是0， 则全部waiting
        #     all_waitting = False
        # # 选出不waiting
        # no_waiting = [1 if waiting == 0 else 0 for waiting in num_reqs_waiting]

        # num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]

        for r in input_requests:
            prefix_lens = [0] * self.dp_size

            with self.recv_tree_cache_lock:
                for gpu_id, radix_cache in self.newest_tree_cache.items():
                    # t_1 = time.time()
                    pre_len = get_match_len(radix_cache.root_node, r.input_ids, 0)
                    # t_2 = time.time()
                    prefix_lens[gpu_id] = pre_len

                # with ThreadPoolExecutor() as executor:
                #     futures = []
                #     for gpu_id, radix_cache in self.newest_tree_cache.items():
                #         future = executor.submit(
                #             self.compute_prefix_length,
                #             gpu_id,
                #             radix_cache,
                #             r.input_ids,
                #         )
                #         futures.append(future)

                #     for future in futures:
                #         gpu_id, pre_len = future.result()
                #         prefix_lens[gpu_id] = pre_len

            # t4 = time.time()
            # with open("match.log", "a+") as f:
            #     f.write(f"[rid={r.rid[:5]}]{prefix_lens}\n")

            # t7 = time.time()
            max_len = max(prefix_lens)
            max_len_indices = [i for i, x in enumerate(prefix_lens) if x == max_len]
            # t8 = time.time()

            # logger.info(f"find max idx = {t8 - t7}")

            if len(max_len_indices) == 1:
                # t9 = time.time()
                selected_worker_index = max_len_indices[0]
                self.workers[selected_worker_index].queue.put(r)
                # t10 = time.time()
                # logger.info(f"len one = {t10 - t9}")
                # t5 = time.time()
                # logger.info(f"if time = {t5 - t4}")
            else:
                self.resources_aware_scheduler([r])
                # t11 = time.time()
                # if all_waitting:
                #     # 全部waiting，选最小的

                #     ratio = [
                #         run / wait
                #         for run, wait in zip(num_reqs_running, num_reqs_waiting)
                #     ]

                #     # run越大 认为后续释放的可能性越多，wait越少，说明后续计算能力更强
                #     min_value = max(ratio)
                #     # 找到所有最小值的索引
                #     min_indices = [i for i, x in enumerate(ratio) if x == min_value]
                #     # 从这些索引中随机选择一个
                #     index = random.choice(min_indices)
                #     # 从waitting最小的找到available最大的
                #     # index = max(min_indices, key=lambda i: available_mem[i])
                #     # index = min(min_indices, key=lambda i: num_reqs_running[i])
                #     self.workers[index].queue.put(r)
                #     num_reqs_waiting[index] += 1
                #     available_mem[index] -= len(r.input_ids)

                # else:
                #     # 选出不waiting的且available mem最大的
                #     # no_waiting 和available做乘法，找最大

                #     filter_result = [a * b for a, b in zip(no_waiting, available_mem)]
                #     index = filter_result.index(max(filter_result))
                #     self.workers[index].queue.put(r)

                #     # num_reqs_running[index] += 1
                #     available_mem[index] -= len(r.input_ids)
                # t12 = time.time()
                # logger.info(f"len two = {t12 - t11}")
                # t5 = time.time()
                # logger.info(f"else time = {t5 - t4}")
            # t6 = time.time()
            # logger.info(f"real dispatch time = {t6 - t8}")

    def resources_aware_scheduler(self, input_requests):
        if len(input_requests) == 0:
            return
        # remained_token = [k.value for k in self.controller_info.waiting_prefill_compute]
        available_mem = [k.value for k in self.controller_info.available_kv_cache]
        num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]
        num_reqs_running = [k.value for k in self.controller_info.running_reqs]
        # with open('three_list.txt', 'a') as file:  # 'a' 模式表示追加到文件末尾
        # print(f"available_mem={available_mem"""  """}\nnum_reqs_waiting={num_reqs_waiting}\nnum_reqs_running={num_reqs_running}\n")

        # ava_resource = available_mem.copy()
        # =======================method2=======================
        # # 认为available + waiting为可用资源
        # for i in range(len(self.workers)):
        #     q = self.workers[i].queue
        #     qsize = q.qsize()
        #     for _ in range(qsize):
        #         req = q.get()
        #         ava_resource[i] += len(req.input_ids)
        #         q.put(req)  # 将元素重新放回原队列

        # # 选择ava最大的调度
        # for r in input_requests:
        #     index = ava_resource.index(max(ava_resource))
        #     self.workers[index].queue.put(r)
        #     ava_resource[index] -= len(r.input_ids)

        # =======================method2=======================

        # =======================method1=======================

        # 判断是否是全部waiting
        all_waitting = False
        if min(num_reqs_waiting) > 0:
            # 最小值都大于0，全部waiting
            all_waitting = True
        else:
            # 最小值都是0， 则全部waiting
            all_waitting = False
        # 选出不waiting
        no_waiting = [1 if waiting == 0 else 0 for waiting in num_reqs_waiting]
        for r in input_requests:
            # t1 = time.time()
            if all_waitting:
                # 全部waiting，选最小的

                ratio = [
                    run / wait for run, wait in zip(num_reqs_running, num_reqs_waiting)
                ]

                # run越大 认为后续释放的可能性越多，wait越少，说明后续计算能力更强
                min_value = max(ratio)
                # 找到所有最小值的索引
                min_indices = [i for i, x in enumerate(ratio) if x == min_value]
                # 从这些索引中随机选择一个
                index = random.choice(min_indices)
                # 从waitting最小的找到available最大的
                # index = max(min_indices, key=lambda i: available_mem[i])
                # index = min(min_indices, key=lambda i: num_reqs_running[i])
                self.workers[index].queue.put(r)
                num_reqs_waiting[index] += 1
                available_mem[index] -= len(r.input_ids)
            else:
                # 选出不waiting的且available mem最大的
                # no_waiting 和available做乘法，找最大

                filter_result = [a * b for a, b in zip(no_waiting, available_mem)]
                index = filter_result.index(max(filter_result))
                self.workers[index].queue.put(r)

                # num_reqs_running[index] += 1
                available_mem[index] -= len(r.input_ids)
            # t2 = time.time()
            # logger.info(f"real dispatch time = {t2 - t1}")

        # =======================method1=======================

    def power_of_2_choice(self, input_requests):
        if len(input_requests) == 0:
            return
        num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]
        num_reqs_running = [k.value for k in self.controller_info.running_reqs]
        available_mem = [k.value for k in self.controller_info.available_kv_cache]

        instances_len = len(self.workers)

        # 比较两个worker的指标
        def compare_metrics(ins1, ins2):
            if num_reqs_waiting[ins1] != num_reqs_waiting[ins2]:
                return ins1 if num_reqs_waiting[ins1] < num_reqs_waiting[ins2] else ins2
            if num_reqs_running[ins1] != num_reqs_running[ins2]:
                return ins1 if num_reqs_running[ins1] < num_reqs_running[ins2] else ins2
            if available_mem[ins1] != available_mem[ins2]:
                return ins1 if available_mem[ins1] > available_mem[ins2] else ins2
            return ins1

        for r in input_requests:
            # 随机选两个worker
            ins1, ins2 = random.sample(range(0, instances_len), 2)
            ins_end = compare_metrics(ins1, ins2)
            self.workers[ins_end].queue.put(r)
            # available_mem[ins_end] -= len(r.input_ids)
            # num_reqs_running[ins_end] += 1
            # num_reqs_waiting[ins_end] += 1

    def round_robin_scheduler(self, input_requests):
        for r in input_requests:
            self.workers[self.round_robin_counter].queue.put(r)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def shortest_queue_scheduler(self, input_requests):
        for r in input_requests:
            queue_sizes = [worker.queue.qsize() for worker in self.workers]
            wid = np.argmin(queue_sizes)
            self.workers[wid].queue.put(r)

    def loop_for_forward(self):
        while True:
            recv_reqs = self.recv_requests()

            if len(recv_reqs) != 0:
                # logger.info(f"len requests=[{len(recv_reqs)}]")
                t1 = time.time()

                # if self.pre_radix:
                #     self.recv_tree_cache()

                self.dispatching(recv_reqs)
                t2 = time.time()
                logger.info(f"scheduler time = {t2 - t1}")

    def loop_for_recv_tree_cache(self):
        while True:
            self.recv_tree_cache()

    def recv_tree_cache(self):
        flag = False
        while True:
            try:
                recv_radix_cache = self.recv_from_tree_cache.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break

            gpu_id = recv_radix_cache.gpu_id

            if (
                gpu_id not in self.newest_tree_cache
                or recv_radix_cache.time > self.newest_tree_cache[gpu_id].time
            ):
                with self.recv_tree_cache_lock:
                    if gpu_id in self.newest_tree_cache:
                        del self.newest_tree_cache[gpu_id]
                    self.newest_tree_cache[gpu_id] = recv_radix_cache
                    flag = True

            del recv_radix_cache
        # 使用日志记录器记录信息
        if flag:
            # logger.info(f"latest_cache={len(self.newest_tree_cache)}")
            pass
        torch.cuda.empty_cache()  # 清空未被引用的显存

    def recv_requests(self):
        recv_reqs = []

        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break

            if isinstance(recv_req, FlushCacheReq):
                # TODO(lsyin): apply more specific flushCacheReq
                for worker in self.workers:
                    worker.queue.put(recv_req)
            elif isinstance(recv_req, AbortReq):
                in_queue = False
                for i, req in enumerate(recv_reqs):
                    if req.rid == recv_req.rid:
                        recv_reqs[i] = recv_req
                        in_queue = True
                        break
                if not in_queue:
                    # Send abort req to all TP groups
                    for worker in self.workers:
                        worker.queue.put(recv_req)
            elif isinstance(recv_req, TokenizedGenerateReqInput):
                recv_reqs.append(recv_req)
            else:
                logger.error(f"Invalid object: {recv_req}")

        return recv_reqs


def start_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    model_overide_args: dict,
):
    """Start a controller process."""

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        controller = ControllerMultiFlex(server_args, port_args, model_overide_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerMultiFlex:\n" + get_exception_traceback())
    finally:
        for w in controller.workers:
            os.kill(w.proc.pid, 9)
        kill_parent_process()
