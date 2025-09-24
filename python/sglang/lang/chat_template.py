import re
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
        template_name = matching_func(model_path)
        if template_name is not None:
            return get_chat_template(template_name)
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

# Reference: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/chat_template.json
register_chat_template(
    ChatTemplate(
        name="mistral",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("[SYSTEM_PROMPT] ", " [/SYSTEM_PROMPT]"),
            "user": ("[INST] ", " [/INST]"),
            "assistant": ("", " </s><s>"),
        },
        stop_str=("</s>",),
        image_token="[IMG]",
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

register_chat_template(
    ChatTemplate(
        name="janus",
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
        image_token="<image_placeholder>\n",
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

# Reference: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/chat_template.json
register_chat_template(
    ChatTemplate(
        name="llama-4",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|header_start|>system<|header_end|>\n\n",
                "<|eot|>",
            ),
            "user": (
                "<|header_start|>user<|header_end|>\n\n",
                "<|eot|>",
            ),
            "assistant": (
                "<|header_start|>assistant<|header_end|>\n\n",
                "<|eot|>",
            ),
        },
        stop_str=("<|eot|>",),
        image_token="<|image|>",
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

# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
register_chat_template(
    ChatTemplate(
        name="internvl-2-5",
        default_system_prompt="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        stop_str=["<|im_end|>", "<|action_end|>"],
    )
)

register_chat_template(
    ChatTemplate(
        name="interns1",
        default_system_prompt="You are an AI assistant whose name is Intern-S1 (书生大模型).\n- Intern-S1 (书生大模型) is a vision-language model that is developed by Shanghai AI Laboratory (上海人工智能实验室).  It is designed to be helpful, honest, and harmless.\n- Intern-S1 (书生大模型) can understand and communicate fluently in the language chosen by the user such as English and 中文.\nYou are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        stop_str=["<|im_end|>", "<|action_end|>"],
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

# Reference: https://huggingface.co/docs/transformers/main/model_doc/glm4_v#usage-example
register_chat_template(
    ChatTemplate(
        name="glm-4v",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<|system|>\n", "\n"),
            "user": ("<|user|>\n", "\n"),
            "assistant": ("<|assistant|>\n", "\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=["<|user|>", "<|endoftext|>", "<|observation|>"],
        image_token="<|image|>",
    )
)


@register_chat_template_matching_function
def match_deepseek(model_path: str):
    if re.search(r"deepseek-(v3|r1)", model_path, re.IGNORECASE) and not re.search(
        r"base", model_path, re.IGNORECASE
    ):
        return "deepseek-v3"


@register_chat_template_matching_function
def match_deepseek_janus_pro(model_path: str):
    if re.search(r"janus", model_path, re.IGNORECASE):
        return "janus-pro"


@register_chat_template_matching_function
def match_dbrx(model_path: str):
    if re.search(r"dbrx", model_path, re.IGNORECASE) and re.search(
        r"instruct", model_path, re.IGNORECASE
    ):
        return "dbrx-instruct"


@register_chat_template_matching_function
def match_vicuna(model_path: str):
    if re.search(r"vicuna|llava-v1\.5|llava-next-video-7b", model_path, re.IGNORECASE):
        return "vicuna_v1.1"


@register_chat_template_matching_function
def match_llama2_chat(model_path: str):
    if re.search(
        r"llama-2.*chat|codellama.*instruct",
        model_path,
        re.IGNORECASE,
    ):
        return "llama-2-chat"


@register_chat_template_matching_function
def match_mistral(model_path: str):
    if re.search(r"pixtral|(mistral|mixtral).*instruct", model_path, re.IGNORECASE):
        return "mistral"


@register_chat_template_matching_function
def match_llama3_instruct(model_path: str):
    if re.search(r"llama-3.*instruct", model_path, re.IGNORECASE):
        return "llama-3-instruct"


@register_chat_template_matching_function
def match_chat_ml(model_path: str):
    if re.search(r"tinyllama", model_path, re.IGNORECASE):
        return "chatml"
    if re.search(r"qwen.*vl", model_path, re.IGNORECASE):
        return "qwen2-vl"
    if re.search(r"glm[-_]?4(\.\d+)?v", model_path, re.IGNORECASE):
        return "glm-4v"
    if re.search(r"qwen.*(chat|instruct)", model_path, re.IGNORECASE) and not re.search(
        r"llava", model_path, re.IGNORECASE
    ):
        return "qwen"
    if re.search(
        r"llava-v1\.6-34b|llava-v1\.6-yi-34b|llava-next-video-34b|llava-onevision-qwen2",
        model_path,
        re.IGNORECASE,
    ):
        return "chatml-llava"


@register_chat_template_matching_function
def match_chat_yi(model_path: str):
    if re.search(r"yi-vl", model_path, re.IGNORECASE) and not re.search(
        r"llava", model_path, re.IGNORECASE
    ):
        return "yi-vl"
    elif re.search(r"yi-1\.5.*chat", model_path, re.IGNORECASE):
        return "yi-1.5"


@register_chat_template_matching_function
def match_gemma_it(model_path: str):
    if re.search(r"gemma.*it", model_path, re.IGNORECASE):
        return "gemma-it"


@register_chat_template_matching_function
def match_openbmb_minicpm(model_path: str):
    if re.search(r"minicpm-v", model_path, re.IGNORECASE):
        return "minicpmv"
    elif re.search(r"minicpm-o", model_path, re.IGNORECASE):
        return "minicpmo"


@register_chat_template_matching_function
def match_c4ai_command_r(model_path: str):
    if re.search(r"c4ai-command-r", model_path, re.IGNORECASE):
        return "c4ai-command-r"


@register_chat_template_matching_function
def match_granite_instruct(model_path: str):
    if re.search(r"granite.*instruct", model_path, re.IGNORECASE):
        return "granite-3-instruct"


@register_chat_template_matching_function
def match_gemma3_instruct(model_path: str):
    if re.search(r"gemma-3", model_path, re.IGNORECASE):
        return "gemma-it"


@register_chat_template_matching_function
def match_internvl_chat(model_path: str):
    if re.search(r"internvl2_5", model_path, re.IGNORECASE):
        return "internvl-2-5"


@register_chat_template_matching_function
def match_interns1_chat(model_path: str):
    if re.search(r"intern-s1", model_path, re.IGNORECASE):
        return "interns1"
    if re.search(r"interns1", model_path, re.IGNORECASE):
        return "interns1"


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
