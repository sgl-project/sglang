import triton_python_backend_utils as pb_utils
import numpy
import sglang as sgl
from sglang import function, set_default_backend
from sglang.srt.constrained import build_regex_from_object

from pydantic import BaseModel

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

class Character(BaseModel):
    name: str
    eye_color: str
    house: str

@function
def character_gen(s, name):
    s += (
        name
        + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    )
    s += sgl.gen("json_output", max_tokens=256, regex=build_regex_from_object(Character))


class TritonPythonModel:
    def initialize(self, args):
        print("Initialized.")
    def execute(self, requests):
        responses = []
        for request in requests:
            tensor_in = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            if tensor_in is None:
                return pb_utils.InferenceResponse(output_tensors=[])
            
            input_list_names = [i.decode('utf-8') if isinstance(i, bytes) else i for i in tensor_in.as_numpy().tolist()]

            input_list_dicts = [{"name":i} for i in input_list_names]

            states = character_gen.run_batch(input_list_dicts)
            character_strs = [state.text() for state in states]

            tensor_out = pb_utils.Tensor("OUTPUT_TEXT", numpy.array(character_strs, dtype=object))

            responses.append(pb_utils.InferenceResponse(output_tensors = [tensor_out]))
        return responses