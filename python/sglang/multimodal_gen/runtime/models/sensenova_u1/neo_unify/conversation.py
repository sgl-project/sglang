"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.

Modified from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

# Modified for SGLang; see this directory's README.md for upstream source.

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.system_message is not None and self.system_message != "":
            system_prompt = self.system_template.format(
                system_message=self.system_message
            )
        else:
            system_prompt = ""

        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = "" if system_prompt == "" else system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = system_prompt if system_prompt != "" else "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            ret = "" if system_prompt == "" else system_prompt + self.sep

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i // 2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + " " + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = "" if system_prompt == "" else self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    if i != len(self.messages) and message is not None:
                        ret += role + self.sep
                    else:
                        ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, (
            f"{template.name} has been registered."
        )

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# Both Hermes-2 and neo1_0-chat are chatml-format conversation templates. The difference
# is that during training, the preprocessing function for the Hermes-2 template doesn't add
# <s> at the beginning of the tokenized sequence, while the neo1_0-chat template does.
# Therefore, they are completely equivalent during inference.

# These Unicode-escaped strings preserve the exact Chinese prompts from the
# pinned upstream source while complying with the multimodal_gen source lint.
# Upstream prompt retained for benchmark compatibility:
_INTERNVL_SYSTEM_MESSAGE = (
    "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4"
    "\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a"
    "\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, "
    "\u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd"
    "\u52a9\u624b\u3002"
)
# Newer upstream prompt, intentionally not used by the first three templates:
_INTERNVL2_5_SYSTEM_MESSAGE = (
    "\u4f60\u662f\u4e66\u751f\xb7\u4e07\u8c61\uff0c\u82f1\u6587\u540d\u662f"
    "InternVL\uff0c\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c"
    "\u5ba4\u3001\u6e05\u534e\u5927\u5b66\u53ca\u591a\u5bb6\u5408\u4f5c\u5355"
    "\u4f4d\u8054\u5408\u5f00\u53d1\u7684\u591a\u6a21\u6001\u5927\u8bed\u8a00"
    "\u6a21\u578b\u3002"
)

register_conv_template(
    Conversation(
        name="Hermes-2",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message=_INTERNVL2_5_SYSTEM_MESSAGE
        system_message=_INTERNVL_SYSTEM_MESSAGE,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_str="<|endoftext|>",
    )
)

register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message=_INTERNVL2_5_SYSTEM_MESSAGE
        system_message=_INTERNVL_SYSTEM_MESSAGE,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
)

register_conv_template(
    Conversation(
        name="phi3-chat",
        system_template="<|system|>\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message=_INTERNVL2_5_SYSTEM_MESSAGE
        system_message=_INTERNVL_SYSTEM_MESSAGE,
        roles=("<|user|>\n", "<|assistant|>\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|end|>",
    )
)

register_conv_template(
    Conversation(
        name="internvl2_5",
        system_template="<|im_start|>system\n{system_message}",
        system_message=_INTERNVL2_5_SYSTEM_MESSAGE,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="neo1_0",
        system_template="<|im_start|>system\n{system_message}",
        system_message="",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)
