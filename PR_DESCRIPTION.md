## PR Motivation

This PR merges the latest changes from the `main` branch into `check-today-issues` to keep the feature branch up-to-date with recent improvements in the multimodal generation testing infrastructure.

The merge includes updates to:
1. **Performance baselines** - Updated performance benchmarks for diffusion generation tests
2. **Test server utilities** - Enhanced test server management and validation capabilities
3. **Test case configurations** - Improved test case definitions and scenario configurations
4. **Test server common functionality** - Better handling of diffusion server lifecycle and test execution

These updates ensure the `check-today-issues` branch benefits from the latest testing improvements and maintains compatibility with the main codebase.

## PR Modifications

### 1. Updated Performance Baselines (`python/sglang/multimodal_gen/test/server/perf_baselines.json`)

- Updated performance benchmark values for diffusion generation test cases
- Adjusted baseline metrics to reflect current expected performance
- Ensures performance regression detection is accurate

### 2. Enhanced Test Server Utilities (`python/sglang/multimodal_gen/test/server/test_server_utils.py`)

**Key Improvements:**
- Added new validation utilities for server performance metrics
- Enhanced `PerformanceValidator` with more robust checking logic
- Improved `ServerManager` for better server lifecycle management
- Added helper functions for test case execution and result validation

### 3. Improved Test Case Configurations (`python/sglang/multimodal_gen/test/server/testcase_configs.py`)

**Changes:**
- Updated `BASELINE_CONFIG` with new performance thresholds
- Enhanced `DiffusionTestCase` with additional configuration options
- Improved `PerformanceSummary` for better metric aggregation
- Updated `ScenarioConfig` to support new test scenarios

### 4. Enhanced Test Server Common (`python/sglang/multimodal_gen/test/server/test_server_common.py`)

**Improvements:**
- Better handling of diffusion server startup and teardown
- Enhanced pytest fixture `diffusion_server` with improved error handling
- Updated server argument construction for new backend options
- Improved compatibility checks for different hardware platforms (AMD/ROCm)

## Files Changed

| File | Changes | Description |
|------|---------|-------------|
| `perf_baselines.json` | 168 lines changed | Updated performance benchmarks |
| `test_server_common.py` | 44 lines changed | Enhanced server test fixtures |
| `test_server_utils.py` | 47 lines changed | Improved test utilities |
| `testcase_configs.py` | 140 lines changed | Updated test configurations |

## Behavior Changes

- **No breaking changes**: This is a merge from main to keep the branch updated
- **Improved test reliability**: Better server management and validation
- **Updated performance expectations**: Baselines reflect current performance

## Backward Compatibility

- **Fully backward compatible**: All changes are additive or updates to test infrastructure
- **No API changes**: Public APIs remain unchanged
- **Test improvements only**: Changes are limited to test files

## Testing

- [x] Verified all test cases pass with updated baselines
- [x] Verified server management utilities work correctly
- [x] Verified performance validation logic is accurate
- [x] Verified compatibility with different hardware platforms
