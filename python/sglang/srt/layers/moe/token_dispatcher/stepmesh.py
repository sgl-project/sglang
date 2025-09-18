import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
import zmq

from sglang.srt.afd.afd_type import afd_is_attn, get_afd_role
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)

dtype_map = [
    torch.float16,  # torch.half
    torch.float32,  # torch.float
    torch.float64,  # torch.double
    torch.bfloat16,  # Brain floating point
    torch.complex32,
    torch.complex64,
    torch.complex128,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,  # torch.int
    torch.int64,  # torch.long
    torch.bool,
    torch.quint8,  # Quantized unsigned int8
    torch.qint8,  # Quantized int8
    torch.qint32,  # Quantized int32
    torch.quint4x2,  # 4-bit quantized (packed)
    torch.quint2x4,  # 2-bit quantized (packed)
]

push_cache = None
initialized = False
fserver = None
global_tokens_num = None


def stepmesh_scheduler():
    os.environ["DMLC_ROLE"] = "scheduler"

    import fserver_lib as f

    f.init()

    # let the scheduler to sleep forever
    while True:
        time.sleep(30 * 24 * 3600)


def env_def(env, v):
    if os.environ.get(env) == None:
        os.environ[env] = v


def start_stepmesh_scheduler(world: GroupCoordinator):
    if os.environ["DMLC_ROLE"] != "server":
        return

    if world.rank_in_group != 0:
        return

    if os.environ.get("STEPMESH_SCHEDULER_STARTED") == "1":
        return

    os.environ["STEPMESH_SCHEDULER_STARTED"] = "1"

    import multiprocessing

    p = multiprocessing.Process(target=stepmesh_scheduler)
    p.daemon = True
    p.start()


def stepmesh_config(world: GroupCoordinator, local_group: GroupCoordinator):
    rank = world.rank_in_group
    local_size = local_group.world_size

    node_rank = rank // local_size

    # Users do not care about these:
    env_def("DMLC_GROUP_SIZE", f"{local_size}")
    env_def("DMLC_NODE_RANK", f"{node_rank}")
    env_def("STEPMESH_GPU", f"{torch.cuda.current_device()}")
    env_def("DMLC_ENABLE_RDMA", "ibverbs")


def att_ffn_exchange_conf(world: GroupCoordinator, local_group: GroupCoordinator):
    rank = world.rank_in_group

    nodes_n = world.world_size // local_group.world_size

    addr = global_server_args_dict.get("afd_ffn_addr")
    ffn_ip, port = addr.split(":")

    port = int(port) + 10

    if rank > 0:
        x = torch.tensor([0])
        # wait that rank0 connects to the peer(attn or ffn)
        x = world.all_gather(x)
    else:
        context = zmq.Context()

        if afd_is_attn():
            assert nodes_n == 1, "Only support one attention node"

            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{ffn_ip}:{port}")

            msg = {"attn_nodes_n": nodes_n, "scheduler_ip": ffn_ip}

            socket.send_string(json.dumps(msg))
            reply = socket.recv_string()

            reply = json.loads(reply)

            peer_nodes_n = reply["ffn_nodes_n"]
        else:
            assert nodes_n == 1, "Only support one FFN node"

            socket = context.socket(zmq.REP)
            socket.bind(f"tcp://*:{port}")

            msg = socket.recv_string()

            msg = json.loads(msg)
            peer_nodes_n = msg["attn_nodes_n"]
            ffn_ip = msg["scheduler_ip"]

            reply = {"ffn_nodes_n": nodes_n}

            socket.send_string(json.dumps(reply))

        x = torch.tensor([peer_nodes_n])
        x = world.all_gather(x)

    peer_nodes_n = x[0]

    if afd_is_attn():
        os.environ["DMLC_NUM_WORKER"] = f"{nodes_n}"
        os.environ["DMLC_NUM_SERVER"] = f"{peer_nodes_n}"
        os.environ["DMLC_ROLE"] = "worker"
    else:
        os.environ["DMLC_NUM_WORKER"] = f"{peer_nodes_n}"
        os.environ["DMLC_NUM_SERVER"] = f"{nodes_n}"
        os.environ["DMLC_ROLE"] = "server"

    os.environ["DMLC_PS_ROOT_URI"] = ffn_ip
    env_def("DMLC_PS_ROOT_PORT", "8123")


def stepmesh_init(local_group: GroupCoordinator):
    global initialized

    if initialized:
        return

    initialized = True

    from sglang.srt.distributed import get_world_group

    world = get_world_group()

    global push_cache
    push_cache = PushTensorCache(world.rank_in_group)

    att_ffn_exchange_conf(world, local_group)
    stepmesh_config(world, local_group)

    import fserver_lib as f

    start_stepmesh_scheduler(world)

    logger.info("stepmesh for %s init..." % get_afd_role())
    f.init()
    logger.info("stepmesh for %s init done." % get_afd_role())

    global fserver
    fserver = f


class StepMeshTensorCache:
    def __init__(self):
        self.push = []
        self.pull = []

        self.push_keys = []
        self.pull_keys = []
        self.shape = None

        self.h = None


class PushTensorCache:
    def __init__(self, rank):
        self.cache = {}
        self.key = rank << 24

    def copy(self, p: PPProxyTensors):
        x = p["hidden_states"]
        topk_weights = p["topk_weights"]
        topk_ids = p["topk_ids"]
        router_logits = p["router_logits"]

        free = self.cache.get(x.shape)
        if free == None:
            self.cache[x.shape] = []
            free = self.cache[x.shape]

        if x.shape[0] == 0:
            l = []
            for t in [x, topk_weights, topk_ids, router_logits]:
                l.append(t.shape[1])
                l.append(dtype_map.index(t.dtype))

            y = torch.tensor(l)

        if len(free) == 0:
            t = StepMeshTensorCache()
            t.shape = x.shape
            t.dtype = x.dtype
            t.device = x.device

            if x.shape[0] == 0:
                x = torch.empty(y.shape[0], dtype=torch.int)
                t.push.append(x)
                t.pull = [torch.empty_like(x)]

            else:
                t.push.append(torch.empty_like(x))
                t.push.append(torch.empty_like(topk_weights))
                t.push.append(torch.empty_like(topk_ids))
                t.push.append(torch.empty_like(router_logits))

                t.pull = [torch.empty_like(x)]

            t.push_keys = range(self.key, self.key + len(t.push))
            self.key += len(t.push)
            t.pull_keys = range(self.key, self.key + len(t.pull))
            self.key += len(t.pull)

        else:
            t = free.pop(0)

        if len(t.push) > 2:
            t.push[0].copy_(x)
            t.push[1].copy_(topk_weights)
            t.push[2].copy_(topk_ids)
            t.push[3].copy_(router_logits)
        else:
            t.push[0].copy_(y)

        return t

    def put(self, t):
        self.cache[t.shape].append(t)


class FFNSendCache:
    def __init__(self):
        self.free_tensors = {}
        self.inflight = []

    def copy(self, x: torch.Tensor):
        if len(self.inflight) > 100:
            t = self.inflight.pop(0)
            self.free_tensors[t.shape].append(t)

        free = self.free_tensors.get(x.shape)
        if free == None:
            self.free_tensors[x.shape] = []
            free = self.free_tensors[x.shape]

        if len(free) == 0:
            t = torch.empty_like(x)
        else:
            t = free.pop(0)

        t.copy_(x)

        self.inflight.append(t)

        return t


ffn_cache = FFNSendCache()


class StepMeshATTN:
    def __init__(
        self,
        layer_id: int,
        group: GroupCoordinator,
    ):
        stepmesh_init(group)

        self.waits = []
        self.f = fserver
        self.layer_id = layer_id

    def send(self, p: PPProxyTensors):

        t = push_cache.copy(p)

        h = self.f.push_pull(t.push, t.push_keys, t.pull, t.pull_keys)

        t.h = h

        self.waits.append(t)

    def recv(self):
        t = self.waits.pop(0)

        self.f.wait(t.h)

        push_cache.put(t)

        if t.shape[0] == 0:
            return torch.empty(0, t.shape[1], dtype=t.dtype, device=t.device)

        return t.pull[0].clone()


def all_gathter_global_tokens(x: torch.Tensor, layer_id: int, group: GroupCoordinator):
    global global_tokens_num
    if layer_id > 0:
        return global_tokens_num

    tensor = torch.tensor([x.shape[0]], dtype=torch.int, device=x.device)

    tensor = group.all_gather(tensor, dim=0)

    global_tokens_num = tensor.tolist()

    return global_tokens_num


class FFNCTX:
    def __init__(self):
        self.id = None
        self.token_n = 0
        self.token_s = 0
        self.token_e = 0
        self.zero_x = None


def all_gather_partial(ctx: FFNCTX, x: torch.Tensor, group: GroupCoordinator):
    dim = x.shape[1]

    t = torch.zeros(ctx.token_n, dim, device=x.device, dtype=x.dtype)
    t[ctx.token_s : ctx.token_e] = x

    torch.ops.sglang.inplace_all_reduce(t, group_name=group.unique_name)

    return t


class StepMeshFFN:
    def __init__(
        self,
        layer_id: int,
        group: GroupCoordinator,
    ):
        stepmesh_init(group)

        self.layer_id = layer_id
        self.comms = []
        self.peek_tensor = None

        self.f = fserver

        self.group = group

    def send(self, x: torch.Tensor):
        x = self.group.all_reduce(x)

        ctx = self.comms.pop(0)

        if ctx.zero_x == None:
            x = x[ctx.token_s : ctx.token_e]

            t = ffn_cache.copy(x)
        else:
            t = ffn_cache.copy(ctx.zero_x)

        self.f.respond([t], ctx.id, True)

    def recv(self):
        batches = self.f.get_batch()
        if len(batches) == 0:
            return None

        from sglang.srt.layers.moe.topk import StandardTopKOutput

        ctx = FFNCTX()
        ctx.id = batches[0][0]

        self.comms.append(ctx)

        # attn send tensor(0, dim)
        if len(batches[0][1]) == 1:
            x = batches[0][1][0]
            ctx.zero_x = x
            y = x.tolist()

            ts = []

            for i in range(0, 4):
                dim = y.pop(0)
                dtype = dtype_map[y.pop(0)]
                ts.append(torch.empty(0, dim, device=x.device, dtype=dtype))

            x, topk_weights, topk_idx, router_logits = ts

        else:
            x, topk_weights, topk_idx, router_logits = batches[0][1]

        g_tokens = all_gathter_global_tokens(x, self.layer_id, self.group)

        ctx.token_s = sum(g_tokens[0 : self.group.rank_in_group])
        ctx.token_e = sum(g_tokens[0 : self.group.rank_in_group + 1])

        if set(g_tokens) == 1:  # tokens number is eq
            x = self.group.all_gather(x, dim=0)
            topk_weights = self.group.all_gather(topk_weights, dim=0)
            topk_idx = self.group.all_gather(topk_idx, dim=0)
            router_logits = self.group.all_gather(router_logits, dim=0)
        else:
            tk = sum(g_tokens)

            ctx.token_n = tk

            x = all_gather_partial(ctx, x, self.group)
            topk_weights = all_gather_partial(ctx, topk_weights, self.group)
            topk_idx = all_gather_partial(ctx, topk_idx, self.group)
            router_logits = all_gather_partial(ctx, router_logits, self.group)

        topk_output = StandardTopKOutput(topk_weights, topk_idx, router_logits)

        return x, topk_output
