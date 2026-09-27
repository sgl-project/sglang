//! Built-in legacy SGLang conversation templates.

use crate::message::types::OneOrMany;

use super::template_legacy::LegacySpec;

pub(super) fn builtin_template(name: &str) -> Option<LegacySpec> {
    let spec = match name {
        "llama-2" => LegacySpec {
            name: name.into(),
            system_template: "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n".into(),
            roles: ("[INST]".into(), "[/INST]".into()),
            style: "LLAMA2".into(),
            sep: " ".into(),
            sep2: Some(" </s><s>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "[INST]".into(),
                "[/INST]".into(),
                "<<SYS>>".into(),
                "<</SYS>>".into(),
            ])),
            ..Default::default()
        },
        "mistral" | "devstral" => LegacySpec {
            name: name.into(),
            system_template: "[SYSTEM_PROMPT]\n{system_message}\n[/SYSTEM_PROMPT]\n\n".into(),
            roles: ("[INST]".into(), "[/INST]".into()),
            style: "LLAMA2".into(),
            sep: " ".into(),
            sep2: Some(" </s><s>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "[INST]".into(),
                "[/INST]".into(),
                "[SYSTEM_PROMPT]".into(),
                "[/SYSTEM_PROMPT]".into(),
            ])),
            ..Default::default()
        },
        "llama-4" => LegacySpec {
            name: name.into(),
            system_template: "<|header_start|>system<|header_end|>\n\n{system_message}<|eot|>"
                .into(),
            roles: ("user".into(), "assistant".into()),
            style: "LLAMA4".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|end_of_text|>".into(),
                "<|eot|>".into(),
                "<|eom|>".into(),
            ])),
            ..Default::default()
        },
        "phi-4-mm" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            roles: ("<|user|>".into(), "<|assistant|>".into()),
            style: "NO_COLON_SINGLE".into(),
            sep: "<|end|>".into(),
            stop_str: Some(OneOrMany::One("<|end|>".into())),
            image_token: "<|endoftext10|>".into(),
            audio_token: "<|endoftext11|>".into(),
            ..Default::default()
        },
        "chatml" | "chatml-llava" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "CHATML".into(),
            sep: "<|im_end|>".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into(),
            ])),
            ..Default::default()
        },
        "vicuna_v1.1" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            system_message: "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.".into(),
            roles: ("USER".into(), "ASSISTANT".into()),
            style: "ADD_COLON_TWO".into(),
            sep: " ".into(),
            sep2: Some("</s>".into()),
            ..Default::default()
        },
        "llama_3_vision" | "llava_llama_3" => LegacySpec {
            name: name.into(),
            system_template: "<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
                .into(),
            system_message: "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.".into(),
            roles: ("user".into(), "assistant".into()),
            style: "LLAMA3".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|end_of_text|>".into(),
                "<|eot_id|>".into(),
            ])),
            ..Default::default()
        },
        "internlm2-chat" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_COLON_SINGLE".into(),
            sep: "\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|action_end|>".into(),
            ])),
            ..Default::default()
        },
        "internvl-2-5" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。".into(),
            roles: ("<|im_start|>user\n".into(), "<|im_start|>assistant\n".into()),
            style: "MPT".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|action_end|>".into(),
            ])),
            ..Default::default()
        },
        "qwen2-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            ..Default::default()
        },
        "deepseek-ocr" => LegacySpec {
            name: name.into(),
            style: "NO_COLON_SINGLE".into(),
            stop_str: Some(OneOrMany::Many(vec!["<｜end▁of▁sentence｜>".into()])),
            ..Default::default()
        },
        "unlimited-ocr" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            style: "UNLIMITED_OCR".into(),
            sep2: Some(String::new()),
            ..Default::default()
        },
        "paddle-ocr" => LegacySpec {
            name: name.into(),
            system_template: "<|begin_of_sentence|>{system_message}".into(),
            roles: ("User".into(), "Assistant".into()),
            style: "PADDLE_OCR".into(),
            sep: "<|end_of_sentence|>".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|end_of_sentence|>".into()])),
            image_token: "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>".into(),
            ..Default::default()
        },
        "deepseek-vl2" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            roles: ("<|User|>".into(), "<|Assistant|>".into()),
            style: "DeepSeekVL2".into(),
            sep: "\n\n".into(),
            sep2: Some("<｜end▁of▁sentence｜>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "User:".into(),
                "<｜end▁of▁sentence｜>".into(),
            ])),
            ..Default::default()
        },
        "gemma-it" => LegacySpec {
            name: name.into(),
            system_template: "<start_of_turn>user\n{system_message}\n\n".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<start_of_turn>user\n".into(), "<start_of_turn>model\n".into()),
            style: "GEMMA3".into(),
            sep: "<end_of_turn>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<end_of_turn>".into()])),
            image_token: "<start_of_image>".into(),
            audio_token: "<start_of_audio>".into(),
            ..Default::default()
        },
        "gme-qwen2-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "QWEN2_VL_EMBED".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::One("<|endoftext|>".into())),
            ..Default::default()
        },
        "minicpmv" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}.".into(),
            system_message: "You are a helpful assistant".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|endoftext|>".into(),
            ])),
            ..Default::default()
        },
        "janus-pro" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}.".into(),
            system_message: "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language".into(),
            roles: ("User".into(), "Assistant".into()),
            style: "ADD_COLON_TWO".into(),
            sep: "\n\n".into(),
            sep2: Some("<｜end▁of▁sentence｜>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "<|User|>".into(),
                "<｜end▁of▁sentence｜>".into(),
            ])),
            ..Default::default()
        },
        "minicpmo" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                .into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|endoftext|>".into(),
            ])),
            ..Default::default()
        },
        "kimi-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_system|>system<|im_middle|>{system_message}".into(),
            system_message: "You are a helpful assistant".into(),
            roles: (
                "<|im_user|>user<|im_middle|>".into(),
                "<|im_assistant|>assistant<|im_middle|>".into(),
            ),
            style: "NO_COLON_SINGLE".into(),
            sep: "<|im_end|>".into(),
            stop_str: Some(OneOrMany::One("<|im_end|>".into())),
            ..Default::default()
        },
        "qwen2-audio" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "QWEN2_AUDIO".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            audio_token: "Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n".into(),
            ..Default::default()
        },
        "moss-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            ..Default::default()
        },
        "points-v15-chat" => LegacySpec {
            name: name.into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            ..Default::default()
        },
        "whisper" => LegacySpec {
            name: name.into(),
            style: "NO_COLON_SINGLE".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|endoftext|>".into()])),
            audio_token: String::new(),
            ..Default::default()
        },
        _ => return None,
    };
    Some(spec)
}
