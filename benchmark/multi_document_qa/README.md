## Run benchmark

### Benchmark sglang
```
python3 -m sglang.launch_server --model-path codellama/CodeLlama-7b-instruct-hf --port 30000
```

```
python3 bench_sglang.py --num-questions 10 --parallel 1
```


### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model codellama/CodeLlama-7b-instruct-hf  --disable-log-requests --port 21000 --gpu 0.97
```

```
python3 bench_other.py --backend vllm --num-questions 64
```


### Benchmark guidance
```
python3 bench_other.py --backend guidance --num-questions 32 --parallel 1 --n-ctx 11000 --model-path path/to/code-llama/gguf
```



### Build dataset

```
pip install PyPDF2
python3 build_dataset.py
```

```python
import PyPDF2

with open('llama2.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    with open('output.txt', 'w') as text_file:
        text_file.write(text)
```
