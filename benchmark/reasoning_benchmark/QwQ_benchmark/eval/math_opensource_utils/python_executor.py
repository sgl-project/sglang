#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import datetime
import io
import logging
import multiprocessing
import pickle
import traceback
from concurrent.futures import TimeoutError
from contextlib import redirect_stdout
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Optional

import dateutil.relativedelta
import regex
from pebble import ProcessPool
from timeout_decorator import timeout
from tqdm import tqdm


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r"(\s|^)?input\(", code_piece):
            # regex.search(r'(\s|^)?os.', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)

        # TODO: use: https://github.com/shroominic/codebox-api
        # @high safe exec in sandbox
        # byte_code = compile_restricted(
        #     code_piece,
        #     filename='<inline code>',
        #     mode='exec'
        # )
        # print("global vars:", self._global_vars)
        # _print_ = PrintCollector
        # exec(byte_code, {'__builtins__': utility_builtins}, None)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars["answer"]


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "datetime": datetime.datetime,
        "timedelta": dateutil.relativedelta.relativedelta,
        "relativedelta": dateutil.relativedelta.relativedelta,
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 5,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.pool = Pool(multiprocessing.cpu_count())
        self.timeout_length = timeout_length

    def process_generation_to_code(self, gens: str):
        return [g.strip().split("\n") for g in gens]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout=None,
        runtime=None,
        answer_symbol=None,
        answer_expr=None,
        timeout_length=10,
        auto_mode=False,
    ):
        try:
            if auto_mode:
                if "print(" in code[-1]:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                    program_io.seek(0)
                    result = program_io.read()
                else:
                    print(code)
                    timeout(timeout_length)(runtime.exec_code)("\n".join(code[:-1]))
                    result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            else:
                if get_answer_from_stdout:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                    program_io.seek(0)
                    result = program_io.read()
                elif answer_symbol:
                    timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                    result = runtime._global_vars[answer_symbol]
                elif answer_expr:
                    timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                    result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
                else:
                    timeout(timeout_length)(runtime.exec_code)("\n".join(code[:-1]))
                    result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            report = "Done"
            str(result)  # codec check
            pickle.dumps(result)  # serialization check
        except:
            result = ""
            report = traceback.format_exc().split("\n")[-2]
        return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        # with ProcessPool(max_workers=min(len(all_code_snippets), os.cpu_count())) as pool:
        with ProcessPool(max_workers=min(len(all_code_snippets), 1)) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length,  # this timeout not work
                auto_mode=True,
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
            else:
                progress_bar = None

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    logging.warning(f"Timeout error in code execution: {error}")
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    raise
                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results


def _test():
    batch_code = [
        """
from sympy import Matrix

def null_space_basis():
    # Define the matrix
    A = Matrix([[3, 3, -1, -6], [9, -1, -8, -1], [7, 4, -2, -9]])

    # Compute the basis for the null space
    basis = A.nullspace()

    # Round the elements of the basis vectors to three decimal places
    basis_rounded = [v.evalf(3) for v in basis]

    return basis_rounded

result = null_space_basis()
print(result)
        """
    ]

    executor = PythonExecutor(get_answer_from_stdout=True)
    predictions = executor.apply(batch_code[0])
    print(predictions)


if __name__ == "__main__":
    _test()
