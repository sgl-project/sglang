use sglang_router_rs::tokenizer::{
    huggingface::HuggingFaceTokenizer,
    traits::{Encoder, Tokenizer},
};

#[tokio::main]
async fn main() {
    // Download Qwen3 tokenizer
    let tokenizer_dir =
        sglang_router_rs::tokenizer::hub::download_tokenizer_from_hf("Qwen/Qwen3-4B-Instruct-2507")
            .await
            .expect("Failed to download tokenizer");

    let tokenizer_path = tokenizer_dir.join("tokenizer.json");

    println!("Loading tokenizer from: {:?}", tokenizer_path);

    let tokenizer =
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).expect("Failed to load");

    let special_tokens = tokenizer.get_special_tokens();

    println!("\n=== Special Tokens ===");
    println!("bos_token: {:?}", special_tokens.bos_token);
    println!("eos_token: {:?}", special_tokens.eos_token);
    println!("unk_token: {:?}", special_tokens.unk_token);
    println!("sep_token: {:?}", special_tokens.sep_token);
    println!("pad_token: {:?}", special_tokens.pad_token);
    println!("cls_token: {:?}", special_tokens.cls_token);
    println!("mask_token: {:?}", special_tokens.mask_token);
    println!(
        "additional_special_tokens: {:?}",
        special_tokens.additional_special_tokens
    );

    // Test encoding with ChatML format
    let test_input =
        "<|im_start|>system\nYou are helpful<|im_end|><|im_start|>user\nHello<|im_end|>";
    println!("\n=== Test Encoding ===");
    println!("Input: {}", test_input);

    let encoding = tokenizer.encode(test_input).expect("Failed to encode");
    println!("Tokens: {:?}", encoding.token_ids());

    // Try to see if the special tokens are in the vocab
    println!("\n=== Token ID Lookup ===");
    println!(
        "<|im_start|> -> {:?}",
        tokenizer.token_to_id("<|im_start|>")
    );
    println!("<|im_end|> -> {:?}", tokenizer.token_to_id("<|im_end|>"));

    // Check vocab with special tokens included
    println!("\n=== Vocab with special=true ===");
    let vocab_with_special = tokenizer.vocab_size(); // Try different methods
    println!("Vocab size (with_special=false): {}", vocab_with_special);

    // Try to find tokens with <| prefix in raw vocab
    println!("\n=== Searching for ChatML-like tokens in vocab ===");
    use sglang_router_rs::tokenizer::traits::Tokenizer as TokenizerTrait;
    let vocab_size = tokenizer.vocab_size();
    println!("Total vocab size: {}", vocab_size);

    // Check specific IDs we know are ChatML tokens
    println!("\nChecking known ChatML token IDs:");
    println!("  ID 151644 -> {:?}", tokenizer.id_to_token(151644));
    println!("  ID 151645 -> {:?}", tokenizer.id_to_token(151645));
}
