# Run Unit Tests

## Test Frontend Language
```
cd sglang/test/lang
export OPENAI_API_KEY=sk-*****

# Run a single file
python3 test_openai_backend.py

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


