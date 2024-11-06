use tokenizers::tokenizer::{Result, Tokenizer};


#[test]
fn test_tokenizer() {
    let tokenizer = Tokenizer::from_file("/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/tokenizer.json").unwrap();
    let encoding = tokenizer.encode("Hey there!", false).unwrap();
    println!("{:?}", encoding.get_ids());
}