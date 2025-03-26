from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple


class ChatTemplateStyle(Enum):
    PLAIN = auto()
    LLAMA2 = auto()


@dataclass
class ChatTemplate:
    name: str
    default_system_prompt: str
    role_prefix_and_suffix: Dict[str, Tuple[str, str]]
    stop_str: List[str] = ()
    image_token: str = "<image>"
    audio_token: str = "<audio>"
    style: ChatTemplateStyle = ChatTemplateStyle.PLAIN

    def get_prefix_and_suffix(
        self, role: str, hist_messages: List[Dict]
    ) -> Tuple[str, str]:
        prefix, suffix = self.role_prefix_and_suffix.get(role, ("", ""))

        if self.style == ChatTemplateStyle.LLAMA2:
            if role == "system" and not hist_messages:
                user_prefix, _ = self.role_prefix_and_suffix.get("user", ("", ""))
                system_prefix, system_suffix = self.role_prefix_and_suffix.get(
                    "system", ("", "")
                )
                return (user_prefix + system_prefix, system_suffix)
            elif (
                role == "user"
                and len(hist_messages) == 1
                and hist_messages[0]["content"] is not None
            ):
                return ("", suffix)

        return prefix, suffix

    def get_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for i, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role == "system" and content is None:
                content = self.default_system_prompt
                if content is None:
                    continue

            prefix, suffix = self.get_prefix_and_suffix(role, messages[:i])
            prompt += f"{prefix}{content}{suffix}"
        return prompt


chat_template_registry: Dict[str, ChatTemplate] = {}
matching_function_registry: List[Callable] = []


def register_chat_template(template):
    chat_template_registry[template.name] = template


def register_chat_template_matching_function(func):
    matching_function_registry.append(func)


def get_chat_template(name):
    return chat_template_registry[name]


def get_chat_template_by_model_path(model_path):
    for matching_func in matching_function_registry:
        template = matching_func(model_path)
        if template is not None:
            return template
    return get_chat_template("default")


register_chat_template(
    ChatTemplate(
        name="default",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("SYSTEM:", "\n"),
            "user": ("USER:", "\n"),
            "assistant": ("ASSISTANT:", "\n"),
        },
    )
)

register_chat_template(
    ChatTemplate(
        name="claude",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("\n\nHuman: ", ""),
            "assistant": ("\n\nAssistant:", ""),
        },
    )
)

register_chat_template(
    ChatTemplate(
        name="chatml",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
    )
)

register_chat_template(
    ChatTemplate(
        name="chatml-llava",
        default_system_prompt="You are a helpful assistant.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
        image_token="<image>\n",
    )
)

# There is default system prompt for qwen
# reference: https://modelscope.cn/models/qwen/Qwen2-72B-Instruct/file/view/master?fileName=tokenizer_config.json&status=1
# The chat template is: "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
register_chat_template(
    ChatTemplate(
        name="qwen",
        default_system_prompt="You are a helpful assistant.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
    )
)

# Reference: https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl#usage-example
register_chat_template(
    ChatTemplate(
        name="qwen2-vl",
        default_system_prompt="You are a helpful assistant.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
    )
)

# Reference: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
register_chat_template(
    ChatTemplate(
        name="vicuna_v1.1",
        default_system_prompt=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
        role_prefix_and_suffix={
            "system": ("", " "),
            "user": ("USER:", " "),
            "assistant": ("ASSISTANT:", "</s>"),
        },
        image_token=" <image>\n",
    )
)

register_chat_template(
    ChatTemplate(
        name="llama-2-chat",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),
            "user": ("[INST] ", " [/INST]"),
            "assistant": ("", " </s><s>"),
        },
        style=ChatTemplateStyle.LLAMA2,
    )
)

register_chat_template(
    ChatTemplate(
        name="llama-3-instruct",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|start_header_id|>system<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "user": (
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "assistant": (
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
        },
        stop_str=("<|eot_id|>",),
        image_token="<|image|>",
    )
)

# https://huggingface.co/openbmb/MiniCPM-V-2_6
register_chat_template(
    ChatTemplate(
        name="minicpmv",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", " "),
            "user": ("user:", " "),
            "assistant": ("assistant:", "</s>"),
        },
        stop_str=("<|im_end|>", "<|endoftext|>"),
        image_token="(<image>./</image>)",
    )
)

register_chat_template(
    ChatTemplate(
        name="janus-pro",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "",
                "",
            ),
            "User": (
                "<｜User｜>",
                "",
            ),
            "assistant": (
                "<｜Assistant｜>",
                "<｜end▁of▁sentence｜>",
            ),
        },
        stop_str=("<｜end▁of▁sentence｜>",),
        image_token="<image_placeholder>\n",
    )
)

# https://huggingface.co/openbmb/MiniCPM-o-2_6
register_chat_template(
    ChatTemplate(
        name="minicpmo",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", " "),
            "user": ("user:", " "),
            "assistant": ("assistant:", "</s>"),
        },
        stop_str=("<|im_end|>", "<|endoftext|>"),
        image_token="(<image>./</image>)",
        audio_token="(<audio>./</audio>)",
    )
)

# The difference between "llama-3-instruct-llava" and "llama-3-instruct" is that llava uses a different image_token.
register_chat_template(
    ChatTemplate(
        name="llama-3-instruct-llava",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|start_header_id|>system<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "user": (
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "assistant": (
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
        },
        stop_str=("<|eot_id|>",),
        image_token="<image>\n",
    )
)

# Reference: https://modelscope.cn/models/01ai/Yi-1.5-34B-Chat/file/view/master?fileName=tokenizer_config.json&status=1
register_chat_template(
    ChatTemplate(
        name="yi-1.5",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
            "assistant": ("", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
    )
)

# Reference: https://github.com/01-ai/Yi/tree/main/VL#major-difference-with-llava
register_chat_template(
    ChatTemplate(
        name="yi-vl",
        default_system_prompt=(
            "This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers."
            "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。"
        ),
        role_prefix_and_suffix={
            "system": ("", "\n\n"),
            "user": ("### Human:", "\n"),
            "assistant": ("### Assistant:", "\n"),
        },
        image_token=" <image_placeholder>\n",
    )
)

register_chat_template(
    ChatTemplate(
        name="gemma-it",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("<start_of_turn>user\n", "<end_of_turn>\n"),
            "assistant": ("<start_of_turn>model\n", "<end_of_turn>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
    )
)

register_chat_template(
    ChatTemplate(
        name="dbrx-instruct",
        default_system_prompt="You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\nYOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\nYou assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\nThis is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>"),
            "user": ("\n<|im_start|>user\n", "<|im_end|>"),
            "assistant": ("\n<|im_start|>assistant\n", "<|im_end|>"),
        },
        stop_str=("<|im_end|>",),
    )
)

register_chat_template(
    ChatTemplate(
        name="c4ai-command-r",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
                "<|END_OF_TURN_TOKEN|>",
            ),
            "user": ("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", "<|END_OF_TURN_TOKEN|>"),
            "assistant": (
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                "<|END_OF_TURN_TOKEN|>",
            ),
        },
        style=ChatTemplateStyle.PLAIN,
    )
)

register_chat_template(
    ChatTemplate(
        name="granite-3-instruct",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|start_of_role|>system<|end_of_role|>",
                "<|end_of_text|>",
            ),
            "user": (
                "<|start_of_role|>user<|end_of_role|>",
                "<|end_of_text|>",
            ),
            "assistant": (
                "<|start_of_role|>assistant<|end_of_role|>",
                "<|end_of_text|>",
            ),
        },
        stop_str=("<|end_of_text|>",),
    )
)

register_chat_template(
    ChatTemplate(
        name="deepseek-v3",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "",
                "",
            ),
            "user": (
                "<｜User｜>",
                "",
            ),
            "assistant": (
                "<｜Assistant｜>",
                "<｜end▁of▁sentence｜>",
            ),
        },
        stop_str=("<｜end▁of▁sentence｜>",),
    )
)


@register_chat_template_matching_function
def match_deepseek(model_path: str):
    if (
        "deepseek-v3" in model_path.lower() or "deepseek-r1" in model_path.lower()
    ) and "base" not in model_path.lower():
        return get_chat_template("deepseek-v3")


@register_chat_template_matching_function
def match_deepseek_janus_pro(model_path: str):
    if "janus" in model_path.lower():
        return get_chat_template("janus-pro")


@register_chat_template_matching_function
def match_dbrx(model_path: str):
    if "dbrx" in model_path.lower() and "instruct" in model_path.lower():
        return get_chat_template("dbrx-instruct")


@register_chat_template_matching_function
def match_vicuna(model_path: str):
    if "vicuna" in model_path.lower():
        return get_chat_template("vicuna_v1.1")
    if "llava-v1.5" in model_path.lower():
        return get_chat_template("vicuna_v1.1")
    if "llava-next-video-7b" in model_path.lower():
        return get_chat_template("vicuna_v1.1")


@register_chat_template_matching_function
def match_llama2_chat(model_path: str):
    model_path = model_path.lower()
    if "llama-2" in model_path and "chat" in model_path:
        return get_chat_template("llama-2-chat")
    if (
        "mistral" in model_path or "mixtral" in model_path
    ) and "instruct" in model_path:
        return get_chat_template("llama-2-chat")
    if "codellama" in model_path and "instruct" in model_path:
        return get_chat_template("llama-2-chat")


@register_chat_template_matching_function
def match_llama3_instruct(model_path: str):
    model_path = model_path.lower()
    if "llama-3" in model_path and "instruct" in model_path:
        return get_chat_template("llama-3-instruct")


@register_chat_template_matching_function
def match_chat_ml(model_path: str):
    # import pdb;pdb.set_trace()
    model_path = model_path.lower()
    if "tinyllama" in model_path:
        return get_chat_template("chatml")
    # Now the suffix for qwen2 chat model is "instruct"
    if "qwen" in model_path and "vl" in model_path:
        return get_chat_template("qwen2-vl")
    if "qwen" in model_path:
        if "vl" in model_path:
            return get_chat_template("qwen2-vl")
        if ("chat" in model_path or "instruct" in model_path) and (
            "llava" not in model_path
        ):
            return get_chat_template("qwen")
    if (
        "llava-v1.6-34b" in model_path
        or "llava-v1.6-yi-34b" in model_path
        or "llava-next-video-34b" in model_path
        or "llava-onevision-qwen2" in model_path
    ):
        return get_chat_template("chatml-llava")


@register_chat_template_matching_function
def match_chat_yi(model_path: str):
    model_path = model_path.lower()
    if "yi-vl" in model_path and "llava" not in model_path:
        return get_chat_template("yi-vl")
    elif "yi-1.5" in model_path and "chat" in model_path:
        return get_chat_template("yi-1.5")


@register_chat_template_matching_function
def match_gemma_it(model_path: str):
    model_path = model_path.lower()
    if "gemma" in model_path and "it" in model_path:
        return get_chat_template("gemma-it")


@register_chat_template_matching_function
def match_openbmb_minicpm(model_path: str):
    model_path = model_path.lower()
    if "minicpm-v" in model_path:
        return get_chat_template("minicpmv")
    elif "minicpm-o" in model_path:
        return get_chat_template("minicpmo")


@register_chat_template_matching_function
def match_c4ai_command_r(model_path: str):
    model_path = model_path.lower()
    if "c4ai-command-r" in model_path:
        return get_chat_template("c4ai-command-r")


@register_chat_template_matching_function
def match_granite_instruct(model_path: str):
    model_path = model_path.lower()
    # When future versions of Granite are released, this code may
    # need to be updated. For now, assume that the Granite 3.0
    # template works across the board.
    if "granite" in model_path and "instruct" in model_path:
        return get_chat_template("granite-3-instruct")


@register_chat_template_matching_function
def match_gemma3_instruct(model_path: str):
    model_path = model_path.lower()
    if "gemma-3" in model_path and "1b" not in model_path:
        # gemma-3-1b-it is completion model
        return get_chat_template("gemma-it")


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": None},  # None means default
        # {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "What can you do?"},
        {"role": "assistant", "content": "I can chat with you."},
    ]

    template = get_chat_template("llama-2-chat")
    print(template.get_prompt(messages))
