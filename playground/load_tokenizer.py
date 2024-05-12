import transformers
import code

#name = "meta-llama/Llama-2-7b-chat-hf"
name = "meta-llama/Meta-Llama-3-8B-Instruct"

t = transformers.AutoTokenizer.from_pretrained(name)
code.interact(local=locals())
