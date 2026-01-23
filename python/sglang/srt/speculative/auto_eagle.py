import math

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory
from typing import Dict, List, Union
import bisect
import json
import os

import logging
logger = logging.getLogger(__name__)


class TuningParams:
    def __init__(self, num_steps: int=5, topk: int=1, num_draft: int=6):
        """todo add tunable param set for spec infer"""
        self.num_steps: int = num_steps
        self.topk: int = topk  # fixed for auto_spec
        self.num_draft: int = num_draft  # num_steps+1 for auto_spec


class AutoTunerEagle:
    def __init__(
        self,
        server_args: ServerArgs,
    ):
        """
        initialise AutoTunerEagle class
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
        self.exp_pos_threshold = {1: 0.55, 2: 0.55, 4: 0.6, 8: 0.8, 16: 0.95, 32: 0.91, 64: 0.95, 128: 0.95}
        self.exp_neg_threshold = {1: 0.5, 2: 0.5, 4: 0.5, 8: 0.5, 16: 0.6, 32: 0.66, 64: 0.65, 128: 0.65}
        # for memory control
        self.step_thres = 6  # control maximum graph number, graph whose num_steps > threshold will be deleted first
        self.reserve_mem = 4
        self.mem_each_graph = 2
        self.save_tune_results = server_args.save_tune_results
        if self.save_tune_results is not None:
            os.makedirs(self.save_tune_results, exist_ok=True)
            self.spec_tune_file = os.path.join(self.save_tune_results, "spec_tune_results.json")
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
        self.max_num_graphs = int((available_memory - self.reserve_mem) // self.mem_each_graph)
        logger.info(f"AutoTunerEagle, max num_graphs to capture for different steps: {self.max_num_graphs}")

        # load speculative config file
        self._init_speculative_config()

        # init original_bs: bs lookup for all bs, since parameters are only recorded for bs in --cuda-graph-bs
        self._generate_closest_bs_mapping()

        # 2. set accept rate to increase or decrease num_steps for different batchsizes
        self.thres_positive_accept_rate = {bs: self.exp_pos_threshold[bs] for bs in self.bs_steps_mapping.keys()}
        self.thres_negative_accept_rate = {bs: self.exp_neg_threshold[bs] for bs in self.bs_steps_mapping.keys()}
        if self.pos_threshold is not None:
            assert len(self.pos_threshold) == len(self.thres_positive_accept_rate)
            for i, bs in enumerate(self.thres_positive_accept_rate.keys()):
                self.thres_positive_accept_rate[bs] = self.pos_threshold[i]
        if self.neg_threshold is not None:
            assert len(self.neg_threshold) == len(self.thres_negative_accept_rate)
            for i, bs in enumerate(self.thres_negative_accept_rate.keys()):
                self.thres_negative_accept_rate[bs] = self.neg_threshold[i]
        logger.info(f"[MY LOG] AutoTunerEagle, pos_thresholds: {self.thres_positive_accept_rate}; neg_thresholds: {self.thres_negative_accept_rate}")

        # 3. initialise recorded speculative parameters
        self.best_speculative_parameters: Dict[int, TuningParams] = {bs: self.exp_setting[bs] for bs in self.bs_list}
        for bs, params in self.best_speculative_parameters.items():
            if params.num_steps not in self.bs_steps_mapping[bs]:
                self.best_speculative_parameters[bs] = TuningParams(self.bs_steps_mapping[bs][-1], 1, self.bs_steps_mapping[bs][-1] + 1)  # if intuitive params not in setting, reset
        logger.info(f"[MY LOG] AutoTunerEagle, best_speculative_parameters init: {self._print_params()}")
        if self.save_tune_results:
            self.results = {bs: {num_steps: 0 for num_steps in self.bs_steps_mapping[bs]} for bs in self.bs_list}

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
                            steps.sort()
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
        if self.cuda_graph_bs != self.bs_list:
            logger.warning(f"AutoTunerEagle, cuda_graph_bs parameter different from speculative config dict, will capture graph according to speculative config.")

        self.bs_range_dict = {}
        for i in range(len(self.bs_list) - 1):
            self.bs_range_dict[self.bs_list[i]] = int((self.bs_list[i] + self.bs_list[i + 1]) // 2)

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

        # deal with scenes when memory available is not enough for the given number of num_steps
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
        search for the closest cuda-graph-bs for the given bs, since the spec parameters are recorded by cuda-graph-bs

        Returns:
            bs in self.bs_list
        """
        # find insert point
        insert_pos = bisect.bisect_left(self.bs_list, target)

        # edge case
        if insert_pos == 0:
            return self.bs_list[0]
        elif insert_pos == len(self.bs_list):
            return self.bs_list[-1]

        # get left and right values
        left_val = self.bs_list[insert_pos - 1]
        right_val = self.bs_list[insert_pos]

        # calculate diff
        left_diff = abs(target - left_val)
        right_diff = abs(target - right_val)

        # choose bs
        if left_diff < right_diff:
            return left_val
        elif right_diff < left_diff:
            return right_val
        else:
            return min(left_val, right_val)

    def _generate_closest_bs_mapping(self) -> dict:
        """
        generate bs: cuda_graph_bs mapping during initialisation to avoid searching in each forward step
        eg: {1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 8, 8: 8, ..., 48: 32, 49: 64, ...}
        """

        start = self.bs_list[0]
        end = self.bs_list[-1]

        self.raw_bs_to_bs = {}

        for i in range(start, end + 1):
            closest = self._find_closest_bs(i)
            if closest is not None:
                self.raw_bs_to_bs[i] = closest

        self.bs_param_fix = {bs: False for bs in self.raw_bs_to_bs.keys()}
        bs_single_param = []
        for bs, num_steps in self.bs_steps_mapping.items():
            if len(num_steps) == 1:
                bs_single_param.append(bs)
        for bs in self.bs_param_fix.keys():
            if self.raw_bs_to_bs[bs] in bs_single_param:
                self.bs_param_fix[bs] = True

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
            while self.best_speculative_parameters[bs].num_steps not in self.bs_steps_mapping[bs]:  # uncontinuous num_steps, eg: {1: [3, 6]}
                self.best_speculative_parameters[bs].num_steps += 1
                self.best_speculative_parameters[bs].num_draft += 1
        else:
            self.best_speculative_parameters[bs].num_steps -= 1
            self.best_speculative_parameters[bs].num_draft -= 1
            while self.best_speculative_parameters[bs].num_steps not in self.bs_steps_mapping[bs]:
                self.best_speculative_parameters[bs].num_steps -= 1
                self.best_speculative_parameters[bs].num_draft -= 1

    def enable_watch_for_batch(self, bs: int):
        if bs not in self.bs_param_fix.keys():
            bs = self.bs_list[-1]
        logger.info(f"[MY LOG] batchsize {bs} enable_watch: {not self.bs_param_fix[bs]}")
        return not self.bs_param_fix[bs]

    def compute_and_update_best_parameters(self, bs, accept_length, accept_rate, throughput):
        """
        calculate best parameters for current batch_size based on negative feedback method and update the
        settings for self.best_speculative_parameters
        """
        # logger.info(f"[MY LOG]AutoTunerEagle compute_and_update_parameters, bs: {bs}, accept_length: {accept_length}, "
        #             f"accept_rate: {accept_rate}, throughput: {throughput}")
        if bs not in self.raw_bs_to_bs.keys():
            bs = self.bs_list[-1]  # exceed max bs, set to max bs
        else:
            bs = self.raw_bs_to_bs[bs]
        # logger.info(f"[MY LOG] AutoTunerEagle bs: {raw_bs}, compute_and_update_parameters round to nearest bs: {bs} ")
        # logger.info(f"[MY LOG]AutoTunerEagle best_speculative_parameters before update: {self._print_params()}")
        accept_rate_pos_flag = accept_rate >= self.thres_positive_accept_rate[bs]
        accept_rate_neg_flag = accept_rate < self.thres_negative_accept_rate[bs]
        # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, bs: {bs}, accept_rate: {accept_rate} vs threshold_neg: {self.thres_negative_accept_rate} vs threshold_pos: {self.thres_positive_accept_rate}")
        if accept_rate_pos_flag:  # increase parameter
            # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, should increase")
            if self.best_speculative_parameters[bs].num_steps < self.bs_steps_mapping[bs][-1]:
                self._update_speculative_params(bs, True)
                if self.save_tune_results:
                    self.results[bs][self.best_speculative_parameters[bs].num_steps] += 1
                return
            # else:
            #     logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, increase abandoned because step out of range")
        elif accept_rate_neg_flag:  # decrease parameter, do not update last parameter
            # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, should decrease")
            if self.best_speculative_parameters[bs].num_steps > self.bs_steps_mapping[bs][0]:
                self._update_speculative_params(bs, False)
                if self.save_tune_results:
                    self.results[bs][self.best_speculative_parameters[bs].num_steps] += 1
                return
            # else:
            #     logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, decrease abandoned because step out of range")
        # otherwise, parameters do not change
        if self.save_tune_results:
            self.results[bs][self.best_speculative_parameters[bs].num_steps] += 1
        # logger.info(f"[MY LOG] AutoTunerEagle compute_and_update_parameters, should remain")
        # logger.info(f"[MY LOG]AutoTunerEagle best_speculative_parameters after update: {self._print_params()}")

    def get_best_parameters(self, bs):
        if bs not in self.raw_bs_to_bs.keys():
            bs = self.bs_list[-1]  # exceed max bs, set to max bs
        else:
            bs = self.raw_bs_to_bs[bs]
        # logger.info(f"[MY LOG]AutoTunerEagle get_best_parameters for bs {bs}, {self._print_params()}")
        return self.best_speculative_parameters[bs]
