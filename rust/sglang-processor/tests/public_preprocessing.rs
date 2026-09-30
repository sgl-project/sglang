use sglang_processor::{ChatTemplateConfig, OneOrMany, load_chat_support};

#[test]
fn built_in_template_is_selected_through_the_public_api() {
    let (formatter, error) = load_chat_support(&ChatTemplateConfig {
        tokenizer_path: ".".into(),
        chat_template: Some("chatml".into()),
        ..Default::default()
    });
    assert!(error.is_none());
    let Some(OneOrMany::Many(stops)) = formatter.unwrap().stop_strs() else {
        panic!("chatml declares multiple stop strings");
    };
    assert_eq!(stops, ["<|endoftext|>", "<|im_end|>"]);
}
