import transformers
import code

name = "meta-llama/Llama-2-7b-chat-hf"

t = transformers.AutoTokenizer.from_pretrained(name)
code.interact(local=locals())
