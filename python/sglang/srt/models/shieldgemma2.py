# Copyright 2025 SGLang Team
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
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from transformers import AutoModel, PreTrainedModel
from transformers.utils.generic import ModelOutput

from sglang.srt.configs import Gemma3Config, ShieldGemma2Config
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.srt.utils import add_prefix

#
# # Copied from: transformers.models.utils
# def is_tensor(x):
#     """
#     Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray`, `np.ndarray` or `mlx.array`
#     in the order defined by `infer_framework_from_repr`
#     """
#     # This gives us a smart order to test the frameworks with the corresponding tests.
#     framework_to_test_func = _get_frameworks_and_test_func(x)
#     for test_func in framework_to_test_func.values():
#         if test_func(x):
#             return True
#
#     # Tracers
#     if is_torch_fx_proxy(x):
#         return True
#
#     if is_flax_available():
#         from jax.core import Tracer
#
#         if isinstance(x, Tracer):
#             return True
#
#     return False

#
# class ModelOutput(OrderedDict):
#     """
#     Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
#     tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
#     python dictionary.
#
#     <Tip warning={true}>
#
#     You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
#     before.
#
#     </Tip>
#     """
#
#     def __init_subclass__(cls) -> None:
#         """Register subclasses as pytree nodes.
#
#         This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
#         `static_graph=True` with modules that output `ModelOutput` subclasses.
#         """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # Subclasses of ModelOutput must use the @dataclass decorator
#         # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
#         # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
#         # Just need to check that the current class is not ModelOutput
#         is_modeloutput_subclass = self.__class__ != ModelOutput
#
#         if is_modeloutput_subclass and not is_dataclass(self):
#             raise TypeError(
#                 f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
#                 " This is a subclass of ModelOutput and so must use the @dataclass decorator."
#             )
#
#     def __post_init__(self):
#         """Check the ModelOutput dataclass.
#
#         Only occurs if @dataclass decorator has been used.
#         """
#         class_fields = fields(self)
#
#         # Safety and consistency checks
#         if not len(class_fields):
#             raise ValueError(f"{self.__class__.__name__} has no fields.")
#         if not all(field.default is None for field in class_fields[1:]):
#             raise ValueError(
#                 f"{self.__class__.__name__} should not have more than one required field."
#             )
#
#         first_field = getattr(self, class_fields[0].name)
#         other_fields_are_none = all(
#             getattr(self, field.name) is None for field in class_fields[1:]
#         )
#
#         if other_fields_are_none and not is_tensor(first_field):
#             if isinstance(first_field, dict):
#                 iterator = first_field.items()
#                 first_field_iterator = True
#             else:
#                 try:
#                     iterator = iter(first_field)
#                     first_field_iterator = True
#                 except TypeError:
#                     first_field_iterator = False
#
#             # if we provided an iterator as first field and the iterator is a (key, value) iterator
#             # set the associated fields
#             if first_field_iterator:
#                 for idx, element in enumerate(iterator):
#                     if (
#                         not isinstance(element, (list, tuple))
#                         or not len(element) == 2
#                         or not isinstance(element[0], str)
#                     ):
#                         if idx == 0:
#                             # If we do not have an iterator of key/values, set it as attribute
#                             self[class_fields[0].name] = first_field
#                         else:
#                             # If we have a mixed iterator, raise an error
#                             raise ValueError(
#                                 f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
#                             )
#                         break
#                     setattr(self, element[0], element[1])
#                     if element[1] is not None:
#                         self[element[0]] = element[1]
#             elif first_field is not None:
#                 self[class_fields[0].name] = first_field
#         else:
#             for field in class_fields:
#                 v = getattr(self, field.name)
#                 if v is not None:
#                     self[field.name] = v
#
#     def __delitem__(self, *args, **kwargs):
#         raise Exception(
#             f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
#         )
#
#     def setdefault(self, *args, **kwargs):
#         raise Exception(
#             f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
#         )
#
#     def pop(self, *args, **kwargs):
#         raise Exception(
#             f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
#         )
#
#     def update(self, *args, **kwargs):
#         raise Exception(
#             f"You cannot use ``update`` on a {self.__class__.__name__} instance."
#         )
#
#     def __getitem__(self, k):
#         if isinstance(k, str):
#             inner_dict = dict(self.items())
#             return inner_dict[k]
#         else:
#             return self.to_tuple()[k]
#
#     def __setattr__(self, name, value):
#         if name in self.keys() and value is not None:
#             # Don't call self.__setitem__ to avoid recursion errors
#             super().__setitem__(name, value)
#         super().__setattr__(name, value)
#
#     def __setitem__(self, key, value):
#         # Will raise a KeyException if needed
#         super().__setitem__(key, value)
#         # Don't call self.__setattr__ to avoid recursion errors
#         super().__setattr__(key, value)
#
#     def __reduce__(self):
#         if not is_dataclass(self):
#             return super().__reduce__()
#         callable, _args, *remaining = super().__reduce__()
#         args = tuple(getattr(self, field.name) for field in fields(self))
#         return callable, args, *remaining
#
#     def to_tuple(self) -> Tuple[Any]:
#         """
#         Convert self to a tuple containing all the attributes/keys that are not `None`.
#         """
#         return tuple(self[k] for k in self.keys())


@dataclass
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ShieldGemma2ImageClassifierOutputWithNoAttention(
    ImageClassifierOutputWithNoAttention
):
    """ShieldGemma2 classifies images as violative or not relative to a specific policy
    Args:
    """

    embeddings: torch.Tensor = None


class ShieldGemma2ForImageClassification(PreTrainedModel):
    config_class = ShieldGemma2Config

    def __init__(
        self,
        config: ShieldGemma2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.yes_token_index = getattr(config, "yes_token_index", 10_784)
        self.no_token_index = getattr(config, "no_token_index", 3771)
        gemma3_config = Gemma3Config(
            text_config=config.text_config,
            vision_config=config.vision_config,
            mm_tokens_per_image=config.mm_tokens_per_image,
            boi_token_index=config.boi_token_index,
            eoi_token_index=config.eoi_token_index,
            image_token_index=config.image_token_index,
            initializer_range=config.initializer_range,
        )
        self.model = Gemma3ForConditionalGeneration(
            config=gemma3_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> ShieldGemma2ImageClassifierOutputWithNoAttention:
        """
        Predicts the binary probability that the image violates the specified policy.

        Args:

        Returns:
        """
        assert (
            get_embedding
        ), "ShieldGemma2ForImageClassification is only used for embedding. Please add --is-embedding when you launch the server."

        out: LogitsProcessorOutput = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        logits = out.next_token_logits
        print(f"logits: {logits}")
        selected_logits = logits[-1, [self.yes_token_index, self.no_token_index]]
        probabilities = torch.softmax(selected_logits, dim=-1)

        return ShieldGemma2ImageClassifierOutputWithNoAttention(
            logits=selected_logits,
            embeddings=probabilities,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        for name, loaded_weight in weights:
            Gemma3ForConditionalGeneration.load_weights(self, [(name, loaded_weight)])


EntryClass = ShieldGemma2ForImageClassification
AutoModel.register(
    config_class=ShieldGemma2Config,
    model_class=ShieldGemma2ForImageClassification,
    exist_ok=True,
)
