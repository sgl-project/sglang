use sglang_processor::{ChatFormatterOptions, OneOrMany, select_chat_formatter};

#[test]
fn built_in_template_is_selected_through_the_public_api() {
    let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
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
