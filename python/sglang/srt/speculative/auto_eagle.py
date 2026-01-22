import math

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory
from typing import Dict, List, Union
import bisect
import json
import os

import logging
logger = logging.getLogger(__name__)

STEP_RANGE = [1, 2, 3, 4, 5, 6]


class TuningParams:
    def __init__(self, num_steps: int=5, topk: int=1, num_draft: int=6):
        """todo add tunable param set for spec infer"""
        self.num_steps: int = num_steps
        self.topk: int = topk  # currently do not tune
        self.num_draft: int = num_draft  # currently do not tune
        self.interval: int = 5  # define the cumulated number of forward to trigger a forward


class AutoTunerEagle:
    def __init__(
        self,
        server_args: ServerArgs,
    ):
        """
        todo
        initialise
        """
        self.cuda_graph_bs = server_args.cuda_graph_bs
        self.device_id = server_args.device
        self.model_name = server_args.served_model_name or server_args.model_path
        self.speculative_config_file = server_args.speculative_config_file
        self.exp_setting = {
            1: TuningParams(5, 1, 6),
            2: TuningParams(5, 1, 6),
            4: TuningParams(5, 1, 6),
            8: TuningParams(4, 1, 5),
            16: TuningParams(3, 1, 4),
            32: TuningParams(3, 1, 4),
            64: TuningParams(3, 1, 4),
            128: TuningParams(1, 1, 2),
        }
        self.step_thres = 6
        self.save_tune_results = server_args.save_tune_results
        self.last_pos_thres = False
        self.last_neg_thres = False
        self.neg_threshold = None
        self.pos_threshold = None
        if server_args.neg_threshold is not None:
            self.neg_threshold = server_args.neg_threshold
        if server_args.pos_threshold is not None:
            self.pos_threshold = server_args.pos_threshold

    def initialize(self, gpu_id: int):
        """must be manually called"""
        # 1. get remaining memory size, determines how many cuda graphs to capture
        available_memory = get_available_gpu_memory(self.device_id, gpu_id, empty_cache=False)
        self.max_num_graphs = int((available_memory - 4) // 2)  # alloc 2GB for each step(verify + draft), 2GB to avoid overflow
        logger.info(f"AutoTunerEagle, num_graphs to capture for different steps: {self.max_num_graphs}")

        # todo: enable config file
        self._init_speculative_config()

        # todo: enable bs lookup for all bs
        self._generate_closest_bs_mapping()

        # 2. calculate the thres_accept_length_growth_rate for each num_steps, need to get the eagle draft model size ratio with respect to target model
        # todo fill in the calculation
        self.thres_accept_length_growth_rate = {step: 0.2 for step in self.step_range}  # todo need calculation
        # self.thres_positive_accept_rate = 0.55
        # self.thres_negative_accept_rate = 0.5

        # todo increase accept rate for larger batchsize
        self.thres_positive_accept_rate = {bs: 0.55 for bs in self.bs_steps_mapping.keys()}
        # self.thres_positive_accept_rate = {bs: 0.6 for bs in self.bs_steps_mapping.keys()}
        self.thres_negative_accept_rate = {bs: 0.5 for bs in self.bs_steps_mapping.keys()}
        # self.thres_negative_accept_rate = {bs: 0.55 for bs in self.bs_steps_mapping.keys()}
        for k in self.thres_positive_accept_rate.keys():
            # if k >= 8 and k <= 32:
            if k == 4:
                self.thres_positive_accept_rate[k] = 0.6
                self.thres_negative_accept_rate[k] = 0.5
                if self.pos_threshold is not None:
                    self.thres_positive_accept_rate[k] = self.pos_threshold[0]
                if self.neg_threshold is not None:
                    self.thres_negative_accept_rate[k] = self.neg_threshold[0]
            if k == 8:
                self.thres_positive_accept_rate[k] = 0.8
                self.thres_negative_accept_rate[k] = 0.5
                if self.pos_threshold is not None:
                    self.thres_positive_accept_rate[k] = self.pos_threshold[1]
                if self.neg_threshold is not None:
                    self.thres_negative_accept_rate[k] = self.neg_threshold[1]
            elif k == 16:
                self.thres_positive_accept_rate[k] = 0.95
                self.thres_negative_accept_rate[k] = 0.6
                if self.pos_threshold is not None:
                    self.thres_positive_accept_rate[k] = self.pos_threshold[2]
                if self.neg_threshold is not None:
                    self.thres_negative_accept_rate[k] = self.neg_threshold[2]
            elif k == 32:
                self.thres_positive_accept_rate[k] = 0.91
                self.thres_negative_accept_rate[k] = 0.66
                if self.pos_threshold is not None:
                    self.thres_positive_accept_rate[k] = self.pos_threshold[3]
                if self.neg_threshold is not None:
                    self.thres_negative_accept_rate[k] = self.neg_threshold[3]
            elif k >= 64:
                self.thres_positive_accept_rate[k] = 0.95
                # self.thres_negative_accept_rate[k] = 0.6
                self.thres_negative_accept_rate[k] = 0.65
                if self.pos_threshold is not None:
                    self.thres_positive_accept_rate[k] = self.pos_threshold[4]
                if self.neg_threshold is not None:
                    self.thres_negative_accept_rate[k] = self.neg_threshold[4]
        logger.info(f"[MY LOG] AutoTunerEagle, pos_thresholds: {self.thres_positive_accept_rate}; neg_thresholds: {self.thres_negative_accept_rate}")

        # 3. initialise recorded speculative parameters
        self.best_speculative_parameters: Dict[int, TuningParams] = {bs: self.exp_setting[bs] for bs in self.cuda_graph_bs}  # todo initialize the best speculative_num_step settings for each bs
        for bs, params in self.best_speculative_parameters.items():
            if params.num_steps not in self.bs_steps_mapping[bs]:
                self.best_speculative_parameters[bs] = TuningParams(self.bs_steps_mapping[bs][-1], 1, self.bs_steps_mapping[bs][-1] + 1)  # if intuitive params not in setting, reset
        # 4. initialise throughput and speculative_num_step recording
        self.last_throughput = {bs: 0. for bs in self.cuda_graph_bs}
        # self.last_accept_length = {bs: 0. for bs in server_args.cuda_graph_bs}
        # self.last_num_steps = {bs: 0 for bs in server_args.cuda_graph_bs}
        self.last_accept_length = {step: 0. for step in self.step_range}
        self.last_time_num_steps_increase_flag = False
        logger.info(f"[MY LOG] AutoTunerEagle, best_speculative_parameters init: {self._print_params()}")
        if self.save_tune_results:
            self.results = {bs: {num_steps: 0 for num_steps in self.bs_steps_mapping[bs]} for bs in self.cuda_graph_bs}

    def _init_speculative_config(self):
        """Initialize speculative configuration from config file or use defaults."""

        # Default configuration
        default_bs_steps_mapping = {
            1: [3, 4, 5, 6],
            2: [3, 4, 5, 6],
            4: [3, 4, 5, 6],
            8: [2, 3, 4],
            16: [2, 3, 4],
            32: [2, 3, 4],
            64: [1, 2, 3, 4],
            128: [1, 2, 3, 4]
        }

        # Check if config file is provided
        if self.speculative_config_file and os.path.exists(self.speculative_config_file):
            try:
                with open(self.speculative_config_file, 'r') as f:
                    config = json.load(f)

                # Get model name from server_args (use model_path as fallback)
                if self.model_name and self.model_name in config:
                    # Use model-specific configuration
                    model_config = config[self.model_name]
                    self.bs_steps_mapping = {}
                    for bs_str, steps in model_config.items():
                        try:
                            bs = int(bs_str)
                            self.bs_steps_mapping[bs] = steps
                        except ValueError:
                            logger.warning(f"[MY LOG] AutoTunerEagle, config init. Invalid batch size in config: {bs_str}")

                    if self.bs_steps_mapping:
                        logger.info(f"[MY LOG] AutoTunerEagle, config init. Loaded speculative config for model '{self.model_name}' from {self.speculative_config_file}, parameters: {self.bs_steps_mapping}")
                    else:
                        logger.warning(f"[MY LOG] AutoTunerEagle, config init. No valid batch size configuration found for model '{self.model_name}', using defaults: {default_bs_steps_mapping}")
                        self.bs_steps_mapping = default_bs_steps_mapping
                else:
                    logger.info(f"[MY LOG] AutoTunerEagle, config init. Model '{self.model_name}' not found in config file, using defaults: {default_bs_steps_mapping}")
                    self.bs_steps_mapping = default_bs_steps_mapping

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[MY LOG] AutoTunerEagle, config init. Failed to load speculative config from {self.speculative_config_file}: {e}, using defaults: {default_bs_steps_mapping}")
                self.bs_steps_mapping = default_bs_steps_mapping
        else:
            # Use default configuration
            self.bs_steps_mapping = default_bs_steps_mapping
            logger.info(f"[MY LOG] AutoTunerEagle, config init. Speculative config file has not been passed, using defaults: {self.bs_steps_mapping}")

        self.step_range = list(set(step for steps in self.bs_steps_mapping.values() for step in steps))
        self.step_range.sort()

        self.bs_list = list(self.bs_steps_mapping.keys())
        self.bs_list.sort()

        # Build reverse mapping: steps -> batch sizes
        self.steps_bs_mapping = {}
        for bs, steps in self.bs_steps_mapping.items():
            for step in steps:
                if step not in self.steps_bs_mapping:
                    self.steps_bs_mapping[step] = []
                self.steps_bs_mapping[step].append(bs)

        # Sort batch sizes for each step
        for step in self.steps_bs_mapping:
            self.steps_bs_mapping[step].sort()

        # todo: deal with scenes when memory available is not enough for the given number of num_steps
        # update self.steps_range  eg: self.step_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.max_num_graphs < len(self.step_range):
            logger.info(f"[MY LOG] AutoTunerEagle init, number of speculative steps to capture: {self.step_range} might exceed available memory.")
            step_range_thres = [step for step in self.step_range if step <= self.step_thres]
            if len(step_range_thres) <= self.max_num_graphs:  # eg: thres=6, max_num_graph=7, remove 8, 9
                self.step_range = self.step_range[:self.max_num_graphs]
            else:  # eg: thres=6, max_num_graph=4
                self.step_range = step_range_thres  # remove 7, 8, 9 first
                temp_step_range = []
                num_even = min(int(math.ceil(len(self.step_range) / 2)), self.max_num_graphs)
                num_odd = self.max_num_graphs - num_even
                for i in range(num_even):
                    temp_step_range.append(self.step_range[2 * i])
                for i in range(num_odd):
                    temp_step_range.append(self.step_range[2 * i + 1])
                self.step_range = temp_step_range
                self.step_range.sort()
            logger.info(f"[MY LOG] AutoTunerEagle init, only capture for steps: {self.step_range}")

            # update self.bs_steps_mapping
            temp_bs_steps_mapping = {}
            for bs, steps in self.bs_steps_mapping.items():
                temp_bs_steps_mapping[bs] = [step for step in steps if step in self.step_range]
                if len(temp_bs_steps_mapping[bs]) == 0:  # if all the steps of given bs is removed, set the bs as the steps_range
                    temp_bs_steps_mapping[bs] = self.step_range
            self.bs_steps_mapping = temp_bs_steps_mapping
            logger.info(f"[MY LOG] AutoTunerEagle init, updated bs_steps_mapping: {self.bs_steps_mapping}")

            # update steps_bs_mapping
            self.steps_bs_mapping = {}
            for bs, steps in self.bs_steps_mapping.items():
                for step in steps:
                    if step not in self.steps_bs_mapping:
                        self.steps_bs_mapping[step] = []
                    self.steps_bs_mapping[step].append(bs)
            logger.info(f"[MY LOG] AutoTunerEagle init, updated steps_bs_mapping: {self.steps_bs_mapping}")

    def _find_closest_bs(self, target: Union[int, float]) -> Union[int, float, None]:
        """
        在有序列表中查找最接近目标值的数

        Args:
            sorted_list: 已排序的数值列表（升序）
            target: 目标数值

        Returns:
            最接近目标值的数，如果列表为空则返回None
        """
        # 使用bisect找到插入位置
        insert_pos = bisect.bisect_left(self.bs_list, target)

        # 处理边界情况
        if insert_pos == 0:
            return self.bs_list[0]
        elif insert_pos == len(self.bs_list):
            return self.bs_list[-1]

        # 获取左右两个候选值
        left_val = self.bs_list[insert_pos - 1]
        right_val = self.bs_list[insert_pos]

        # 计算差值
        left_diff = abs(target - left_val)
        right_diff = abs(target - right_val)

        # 根据差值选择结果，差值相同时取较小的数
        if left_diff < right_diff:
            return left_val
        elif right_diff < left_diff:
            return right_val
        else:
            return min(left_val, right_val)

    def _generate_closest_bs_mapping(self) -> dict:
        """
        优化的整数到最接近值的映射字典生成函数

        使用更高效的方法生成映射，避免对每个整数都进行二分查找

        Args:
            sorted_list: 已排序的数值列表（升序）

        Returns:
            字典，key为范围内的整数，value为最接近的列表中的数
        """

        start = self.bs_list[0]
        end = self.bs_list[-1]

        self.raw_bs_to_bs = {}

        # 使用双指针方法优化
        for i in range(start, end + 1):
            # 使用二分查找找到最接近的值
            closest = self._find_closest_bs(i)
            if closest is not None:
                self.raw_bs_to_bs[i] = closest

    def _print_params(self, bs: int = None):
        msg = ""
        if bs:
            params = self.best_speculative_parameters[bs]
            return f"{bs}:{params.num_steps} {params.topk} {params.num_draft}"
        for bs, params in self.best_speculative_parameters.items():
            msg += f"{bs}:{params.num_steps} {params.topk} {params.num_draft}\t"
        return msg

    def _update_speculative_params(self, bs: int, increase: bool = True):
        # logger.info(f"[MY LOGS] AutoTunerEagle._update_speculative_params, bs={bs}, "
        #             f"best_speculative_parameters={self.best_speculative_parameters}"
        #             f"bs_steps_mapping={self.bs_steps_mapping}")
        if increase:
            self.best_speculative_parameters[bs].num_steps += 1
            self.best_speculative_parameters[bs].num_draft += 1
            while self.best_speculative_parameters[bs].num_steps not in self.bs_steps_mapping[bs]:  # scene when there's no neighbouring num_steps
                self.best_speculative_parameters[bs].num_steps += 1
                self.best_speculative_parameters[bs].num_draft += 1
            # for b in self.best_speculative_parameters.keys():
            #     if b == bs:
            #         continue
            #     if self.best_speculative_parameters[bs].num_steps <= self.bs_steps_mapping[b][-1]:
            #         self.best_speculative_parameters[b].num_steps = self.best_speculative_parameters[bs].num_steps
            #         self.best_speculative_parameters[b].num_draft = self.best_speculative_parameters[bs].num_draft
        else:
            self.best_speculative_parameters[bs].num_steps -= 1
            self.best_speculative_parameters[bs].num_draft -= 1
            while self.best_speculative_parameters[bs].num_steps not in self.bs_steps_mapping[bs]:  # scene when there's no neighbouring num_steps
                self.best_speculative_parameters[bs].num_steps -= 1
                self.best_speculative_parameters[bs].num_draft -= 1
            # for b in self.best_speculative_parameters.keys():
            #     if b == bs:
            #         continue
            #     if self.best_speculative_parameters[bs].num_steps >= self.bs_steps_mapping[b][0]:
            #         self.best_speculative_parameters[b].num_steps = self.best_speculative_parameters[bs].num_steps
            #         self.best_speculative_parameters[b].num_draft = self.best_speculative_parameters[bs].num_draft

    def _update_metrics(self, bs: int, num_steps: int, accept_length, throughput):
        self.last_accept_length[num_steps] = accept_length
        self.last_throughput[bs] = throughput
        # logger.info(f"[MY LOG] AutoTunerEagle.update_metrics, num_steps: {num_steps}, self.last_accept_length: {self.last_accept_length[num_steps]:.2f}, "
        #             f"bs: {bs}, self.last_throughput: {self.last_throughput[bs]:.2f}")

    def compute_batchsize32(self, current_num_steps: int, accept_length: int):
        if current_num_steps > 3:
            self.best_speculative_parameters[32].num_steps = 3
            self.best_speculative_parameters[32].num_draft = 4
            self.best_speculative_parameters[32].topk = 1
            return
        # 参数2
        if current_num_steps == 3 and accept_length < 2.2:
            self._update_speculative_params(32, False)
        elif current_num_steps == 2:
            if accept_length < 1.9:
                self._update_speculative_params(32, False)
            elif accept_length > 2.1:
                self._update_speculative_params(32, True)
        elif current_num_steps == 1:
            if accept_length > 1.7:
                self._update_speculative_params(32, True)
        return
        # # 参数3
        # self.best_speculative_parameters[32].num_steps = 3
        # self.best_speculative_parameters[32].num_draft = 4
        # self.best_speculative_parameters[32].topk = 1

    def compute_and_update_best_parameters(self, bs, accept_length, accept_rate, throughput):
        """
        todo special analysis for bs=32
        calculate best parameters for current batch_size based on negative feedback method and update the
        settings for self.best_speculative_parameters
        """
        # logger.info(f"[MY LOG]AutoTunerEagle compute_and_update_parameters, bs: {bs}, accept_length: {accept_length}, "
        #             f"accept_rate: {accept_rate}, throughput: {throughput}")
        raw_bs = bs
        if bs not in self.raw_bs_to_bs.keys():
            bs = self.bs_list[-1]  # exceed max bs, set to max bs
        else:
            bs = self.raw_bs_to_bs[bs]
        # logger.info(f"[MY LOG] AutoTunerEagle bs: {raw_bs}, compute_and_update_parameters round to nearest bs: {bs} ")
        # logger.info(f"[MY LOG]AutoTunerEagle best_speculative_parameters before update: {self._print_params()}")
        current_num_steps = self.best_speculative_parameters[bs].num_steps  # 假设当前num_steps=2
        small_compare_num_steps = current_num_steps - 1  # 则应该和num_steps=1是的接受长度比增长率
        accept_length_flag = 1
        # todo bs=1/2特殊处理，连续两次才触发决策
        # todo bs=32的特殊处理，但是对于不同的部署形态不能通用
        # if bs == 32:
        #     self.compute_batchsize32(current_num_steps, accept_length)
        #     return
        # todo 先取消growth length规则
        # if self.last_time_num_steps_increase_flag:  # 有上次的记录, 且当前的num_steps相较于上次是增加的, 否则计算这个参数没有意义; last_num_steps>=current_num_steps，都只看accept_rate
        #     logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, find metric record")
        #     last_accept_length = self.last_accept_length[small_compare_num_steps]
        #     accept_length_growth_rate = (accept_length - last_accept_length) / last_accept_length
        #     accept_length_flag = accept_length_growth_rate >= self.thres_accept_length_growth_rate[small_compare_num_steps]
        #     logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, bs: {bs}, "
        #                 f"accept_length_growth_rate: {accept_length_growth_rate} vs threshold: {self.thres_accept_length_growth_rate[current_num_steps - 1]}")
        accept_rate_pos_flag = accept_rate >= self.thres_positive_accept_rate[bs]
        accept_rate_neg_flag = accept_rate < self.thres_negative_accept_rate[bs]
        # if bs == 1 or bs == 2:  # bs=1或2时，连续两次触发阈值才调整参数
        #     if accept_rate_pos_flag and (not self.last_pos_thres):
        #         # 第一次触发pos
        #         # logger.info(f"[MY LOG] bs={bs} 1st exceeds pos thres")
        #         accept_rate_pos_flag = False
        #         self.last_pos_thres = True
        #     elif accept_rate_pos_flag and self.last_pos_thres:
        #         # 第二次触发pos
        #         # logger.info(f"[MY LOG] bs={bs} 2nd exceeds pos thres")
        #         self.last_pos_thres = False
        #     elif accept_rate_neg_flag and (not self.last_neg_thres):
        #         # 第一次触发neg
        #         # logger.info(f"[MY LOG] bs={bs} 1st exceeds neg thres")
        #         accept_rate_neg_flag = False
        #         self.last_neg_thres = True
        #     elif accept_rate_neg_flag and self.last_neg_thres:
        #         # 第二次触发neg
        #         # logger.info(f"[MY LOG] bs={bs} 2nd exceeds neg thres")
        #         self.last_neg_thres = False
        #     elif self.last_pos_thres:
        #         # 第一次触发，第二次没触发，置0
        #         # logger.info(f"[MY LOG] bs={bs} 2nd didn't exceeds pos thres")
        #         self.last_pos_thres = False
        #     elif self.last_neg_thres:
        #         # 第一次触发，第二次没触发，置0
        #         # logger.info(f"[MY LOG] bs={bs} 2nd didn't exceeds neg thres")
        #         self.last_neg_thres = False
        # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, bs: {bs}, accept_rate: {accept_rate} vs threshold_neg: {self.thres_negative_accept_rate} vs threshold_pos: {self.thres_positive_accept_rate}")
        self._update_metrics(bs, current_num_steps, accept_length, throughput)
        if accept_length_flag and accept_rate_pos_flag:  # increase parameter
            # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, should increase")
            if ((self.best_speculative_parameters[bs].num_steps + 1) in self.bs_steps_mapping[bs]) or ((self.best_speculative_parameters[bs].num_steps + 2) in self.bs_steps_mapping[bs]) or ((self.best_speculative_parameters[bs].num_steps + 3) in self.bs_steps_mapping[bs]):
                self._update_speculative_params(bs, True)
                self.last_time_num_steps_increase_flag = True
                if self.save_tune_results:
                    self.results[bs][self.best_speculative_parameters[bs].num_steps] += 1
                return
            # else:
            #     logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, increase abandoned because step out of range")
        elif (not accept_length_flag) or accept_rate_neg_flag:  # decrease parameter, do not update last parameter
            # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, should decrease")
            if ((self.best_speculative_parameters[bs].num_steps - 1) in self.bs_steps_mapping[bs]) or ((self.best_speculative_parameters[bs].num_steps - 2) in self.bs_steps_mapping[bs]) or ((self.best_speculative_parameters[bs].num_steps - 3) in self.bs_steps_mapping[bs]):
                self.last_time_num_steps_increase_flag = False
                self._update_speculative_params(bs, False)
                if self.save_tune_results:
                    self.results[bs][self.best_speculative_parameters[bs].num_steps] += 1
                return
            # else:
            #     logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, decrease abandoned because step out of range")
        # otherwise, parameters do not change
        self.last_time_num_steps_increase_flag = False
        if self.save_tune_results:
            self.results[bs][self.best_speculative_parameters[bs].num_steps] += 1
        # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, should remain")
        # logger.info(f"[MY LOG]AutoTunerEagle best_speculative_parameters after update: {self._print_params()}")

    def get_best_parameters(self, bs):
        # todo mock for debug
        if bs not in self.raw_bs_to_bs.keys():
            bs = self.bs_list[-1]  # exceed max bs, set to max bs
        else:
            bs = self.raw_bs_to_bs[bs]
        # todo mock for debug
        # logger.info(f"[MY LOG]AutoTunerEagle get_best_parameters for bs {bs}, {self._print_params()}")
        return self.best_speculative_parameters[bs]
