# Run Unit Tests

SGLang uses the built-in library [unittest](https://docs.python.org/3/library/unittest.html) as the testing framework.  

## Test Frontend Language
```
cd sglang/test/lang
export OPENAI_API_KEY=sk-*****

# Run a single file
python3 test_openai_backend.py

# Run a single test
python3 -m unittest test_openai_backend.TestOpenAIBackend.test_few_shot_qa

# Run a suite
python3 run_suite.py --suite minimal
```

## Test Backend Runtime
```
cd sglang/test/srt

# Run a single file
python3 test_eval_accuracy.py

# Run a suite
python3 run_suite.py --suite minimal
```
