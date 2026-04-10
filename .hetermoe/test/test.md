ALL the simplest working unit of our implementation should be tested thoroughly. 
try to come up with tests that touches:
    1. usability
    2. accuracy
    3. efficiency
    4. corner case

some tests can be constructed with fake weights(layers) and fake inputs refer to plan.md for designing the tests
other tests requries the use of real model weights and inputs... reduce the frequency of these large tests

you should make yourself guidelines for both unit tests and e2e integration tests in this folder!
these two usually have different metrics for accuracy, for example, unit tests with fake inputs simply needs to compute MSE for accuracy
    however, e2e integration tests require either simple perplexity or to run lm-eval for downstream scoring

efficiency tests should be tested with
    1. torch compile
    2. cudagraph
    3. warm up
    4. a large enough working set to avoid L2 camping [IMPORTANT]

tests are prefered in pytest