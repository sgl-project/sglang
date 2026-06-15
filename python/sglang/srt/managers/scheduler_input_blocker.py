# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, List, Optional

from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType
from sglang.srt.utils.poll_based_barrier import PollBasedBarrier

logger = logging.getLogger(__name__)


class SchedulerInputBlocker:
    """
    SchedulerInputBlocker 用于在 Scheduler 端阻断和缓存输入请求。
    配合 Tokenizer 端的 input_blocker_guard_region 上下文管理器使用，
    实现大 Batch 请求的‘原子打包发送’与‘分布式同步释放’。
    """
    def __init__(self, noop: bool):
        # 初始状态为未阻断
        self._state = _State.UNBLOCKED
        # 用于缓存被阻断期间到达的普通请求
        self._pending_reqs = []
        # noop=True 时为“空操作”模式（例如在不需要多卡同步、也不开启Colocated Batch的单卡/简单场景下）
        self._noop = noop
        # 分布式全局同步屏障（基于轮询，避免阻塞主推理线程）
        self._global_unblock_barrier = PollBasedBarrier(noop=noop)

    def handle(self, recv_reqs: Optional[List[Any]]):
        """
        每轮调度循环中调用的主入口。
        处理新接收到的请求，并轮询全局同步屏障状态。
        """
        # 如果是 noop 模式，接收到的 reqs 应该为 None，反之亦然
        assert (recv_reqs is None) == self._noop

        if not self._noop:
            output_reqs = []
            # 1. 逐个处理当前轮次收到的请求/控制信号
            for recv_req in recv_reqs:
                output_reqs += self._handle_recv_req(recv_req)

        # 2. 轮询分布式全局同步屏障，检查其他所有的 Rank 是否也已经到达了解锁屏障
        global_arrived_unblock_barrier = (
            self._global_unblock_barrier.poll_global_arrived()
        )
        # 3. 如果当前处于“等待全局解锁屏障”状态，且所有 Rank 都已到达
        if (
            self._state == _State.GLOBAL_UNBLOCK_BARRIER
            and global_arrived_unblock_barrier
        ):
            # 释放所有积攒的缓存请求，并恢复到 UNBLOCKED 状态
            output_reqs += self._handle_arrive_unblock_barrier()

        if not self._noop:
            return output_reqs

    def _handle_recv_req(self, recv_req):
        """
        处理单个请求（区分控制请求 BlockReqInput 和普通的推理请求）
        """
        if isinstance(recv_req, BlockReqInput):
            # 处理控制阻断/解锁的特殊请求
            if recv_req.type == BlockReqType.BLOCK:
                self._execute_block_req()
                return []
            elif recv_req.type == BlockReqType.UNBLOCK:
                self._execute_unblock_req()
                return []
            else:
                raise NotImplementedError(f"{recv_req=}")
        else:
            # 处理普通推理请求
            if self._state == _State.UNBLOCKED:
                # 未阻断状态：直接把请求透传出去进行调度
                return [recv_req]
            else:
                # 阻断状态（BLOCKED 或 GLOBAL_UNBLOCK_BARRIER）：
                # 将请求存入临时缓存区，不立刻调度
                self._pending_reqs.append(recv_req)
                return []

    def _execute_block_req(self):
        """收到 BLOCK 信号：从 UNBLOCKED 切换到 BLOCKED 状态"""
        logger.info("Handle block req")
        self._change_state(original=_State.UNBLOCKED, target=_State.BLOCKED)

    def _execute_unblock_req(self):
        """收到 UNBLOCK 信号：从 BLOCKED 切换到 GLOBAL_UNBLOCK_BARRIER 状态，并声明本地已到达屏障"""
        logger.info("Handle unblock req")
        self._change_state(
            original=_State.BLOCKED, target=_State.GLOBAL_UNBLOCK_BARRIER
        )
        # 告知屏障管理器：本 Rank 已经接收完了所有的请求
        self._global_unblock_barrier.local_arrive()

    def _handle_arrive_unblock_barrier(self):
        """所有分布式节点全部完成了接收：解锁，排空并返回所有缓存的推理请求"""
        logger.info(f"Arrived at unblock barrier ({len(self._pending_reqs)=})")
        # 状态重置为 UNBLOCKED
        self._change_state(
            original=_State.GLOBAL_UNBLOCK_BARRIER, target=_State.UNBLOCKED
        )
        # 复制并清空缓存队列
        output_reqs = [*self._pending_reqs]
        self._pending_reqs.clear()
        return output_reqs

    def _change_state(self, original: "_State", target: "_State"):
        """状态断言切换，确保状态机流转符合预期，防止时序错乱"""
        assert self._state == original, f"{self._state=} {original=} {target=}"
        self._state = target


class _State(Enum):
    UNBLOCKED = auto()  # 正常运行，不阻断
    BLOCKED = auto()  # 阻断中，缓存所有推理请求
    GLOBAL_UNBLOCK_BARRIER = auto()  # 收到 UNBLOCK 后，等待分布式多卡对齐同步的状态


@contextmanager
def input_blocker_guard_region(send_to_scheduler):
    """
    Tokenizer 端使用的上下文管理器。
    用于包裹一整批请求的分词和发送过程，确保在这个区域（Region）内，Scheduler 维持在阻断状态。
    """
    # 1. 进入 with 块：立刻向 Scheduler 发送 BLOCK 控制信号
    send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.BLOCK))
    try:
        yield
    finally:
        # 2. 退出 with 块（即使中途发生异常）：向 Scheduler 发送 UNBLOCK 信号
        send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.UNBLOCK))
