from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs


class SchedulerRecvSkipper:
    """
    SchedulerRecvSkipper 用于控制模型执行器 (Executor) 轮询/接收调度器 (Scheduler) 新请求的频率。
    
    主要目的是在推理过程中（特别是耗时极短的 DECODE 阶段），避免每一步都进行通信去获取请求，
    从而减少系统的 IPC 开销，提升推理服务的吞吐量与降低延迟。
    """
    @staticmethod
    def maybe_create(server_args: ServerArgs):
        """
        静态工厂方法：根据服务启动参数决定是否创建 Skipper 实例。
        如果 scheduler_recv_interval <= 1（默认是 1），表示每步都接收，不进行跳过，此时不创建 Skipper，返回 None。
        """
        if server_args.scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper(server_args)

    def __init__(self, server_args: ServerArgs):
        # 数据并行 (Data Parallel) Attention 场景暂不支持跳过机制（因为 DP 场景通常需要严格的每步同步）
        # Can be supported if needed, but may need e.g. `global_forward_mode`
        assert not server_args.enable_dp_attention
        # 内部权重计数器，用于累积每一步的权重
        self._counter = 0
        # 触发接收的阈值，对应启动参数 --scheduler-recv-interval 的值
        self._threshold = server_args.scheduler_recv_interval
        # All can be tuned if needed
        # 默认权重：当 ForwardMode 无法匹配到字典中的特定模式时（如 PREFILL 阶段或其它未知阶段）使用。
        # 默认值从环境变量 SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT 中读取（通常是一个较大的值，如 1000，以确保立刻触发接收）
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        # 针对不同的前向传播模式 (ForwardMode) 定义每次累加的权重：
        # - DECODE: 生成 Token 阶段，单步极快，权重通常设得很小（如 1），需要累加多次才触发一次接收。
        # - TARGET_VERIFY: 投机采样验证阶段，同样较快，权重通常也设得很小（如 1）。
        # - None: 未指定模式，权重由环境变量配置。
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def handle(self, last_forward_mode: ForwardMode):
        """
        处理单步推理结束后的状态更新。
        根据上一步的前向传播模式增加计数器，并判断当前步是否应该接收 (recv) 新的调度器请求。
        
        :param last_forward_mode: 上一次前向传播的模式 (例如 DECODE, TARGET_VERIFY 等)
        :return: bool，True 表示当前步应该向调度器接收新请求，False 表示跳过接收
        """
        should_recv = False

        # 1. 获取当前模式对应的权重。如果模式不在字典中，则使用默认的 self._default_weight
        last_weight = self._weight_of_forward_mode.get(
            last_forward_mode, self._default_weight
        )

        # 2. 累加权重到计数器中
        self._counter += last_weight

        # 3. 如果累计权重达到或超过阈值，重置计数器并标记为应当接收新请求
        if self._counter >= self._threshold:
            self._counter = 0
            should_recv = True

        return should_recv
