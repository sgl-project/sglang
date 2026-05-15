import os
import sys
import time
import traceback
from pathlib import Path

from sglang.benchmark.backends.base_client import (
    BaseBackendClient,
    RequestFuncInput,
    RequestFuncOutput,
)
from sglang.benchmark.backends.common import _create_bench_client_session


class ProfileBackendClient(BaseBackendClient):
    async def request(
        self,
        request_func_input: RequestFuncInput,
        pbar=None,
    ) -> RequestFuncOutput:
        raise NotImplementedError("ProfileBackendClient does not handle data requests")

    async def request_profile(self, api_url: str) -> RequestFuncOutput:
        async with _create_bench_client_session() as session:
            output = RequestFuncOutput()
            try:
                if api_url.endswith("/start_profile"):
                    num_steps = getattr(self.args, "profile_num_steps", None)
                    profile_by_stage = getattr(self.args, "profile_by_stage", None)
                    if profile_by_stage and num_steps is None:
                        num_steps = 5

                    output_dir = getattr(self.args, "profile_output_dir", None)
                    if output_dir is None:
                        output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
                    output_dir = Path(
                        os.path.abspath(os.path.normpath(output_dir))
                    ) / str(time.time())
                    output_dir.mkdir(exist_ok=True, parents=True)
                    output_dir = str(output_dir)

                    body = {
                        "activities": getattr(self.args, "profile_activities", []),
                        "num_steps": num_steps,
                        "profile_by_stage": profile_by_stage,
                        "profile_stages": getattr(self.args, "profile_stages", None),
                        "output_dir": output_dir,
                        "profile_prefix": getattr(self.args, "profile_prefix", None),
                    }
                else:
                    # stop_profile doesn't need any parameters
                    body = {}
                # Add optional profiling parameters if provided.
                if (
                    hasattr(self.args, "profile_start_step")
                    and self.args.profile_start_step is not None
                ):
                    body["start_step"] = str(self.args.profile_start_step)
                if (
                    hasattr(self.args, "profile_steps")
                    and self.args.profile_steps is not None
                ):
                    body["num_steps"] = str(self.args.profile_steps)
                async with session.post(url=api_url, json=body) as response:
                    if response.status == 200:
                        output.success = True
                    else:
                        output.error = (
                            (response.reason or "") + ": " + (await response.text())
                        )
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        return output
