# SGLang CI Monitor

A simple tool to analyze CI failures for the SGLang project. This tool fetches recent CI run data from GitHub Actions and provides detailed analysis of failure patterns.

## Features

- **Simple Analysis**: Analyze recent CI runs and identify failure patterns
- **Category Classification**: Automatically categorize failures by type (unit-test, performance, etc.)
- **Pattern Recognition**: Identify common failure patterns (timeouts, build failures, etc.)
- **CI Links**: Direct links to recent failed CI runs for detailed investigation
- **JSON Export**: Export detailed analysis data to JSON format

## Installation

No additional dependencies required beyond Python standard library and `requests`:

```bash
pip install requests
```

## Usage

### Basic Usage

```bash
# Replace YOUR_GITHUB_TOKEN with your actual token from https://github.com/settings/tokens
python ci_analyzer.py --token YOUR_GITHUB_TOKEN
```

### Advanced Usage

```bash
# Analyze last 1000 runs
python ci_analyzer.py --token YOUR_GITHUB_TOKEN --limit 1000

# Custom output file
python ci_analyzer.py --token YOUR_GITHUB_TOKEN --limit 500 --output my_analysis.json
```

**Important**: Make sure your GitHub token has `repo` and `workflow` permissions, otherwise you'll get 404 errors.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--token` | Required | GitHub Personal Access Token |
| `--limit` | 100 | Number of CI runs to analyze |
| `--output` | ci_analysis.json | Output JSON file for detailed data |

## Getting GitHub Token

1. Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token" > "Generate new token (classic)"
3. **Important**: Select the following permissions:
   - `repo` (Full control of private repositories) - **Required for accessing repository data**
   - `workflow` (Update GitHub Action workflows) - **Required for reading CI/CD data**
4. Copy the generated token and use it as `YOUR_GITHUB_TOKEN`

**Note**: Without the `repo` and `workflow` permissions, the tool will not be able to access CI run data and will return 404 errors.

## Output

The tool provides:

### Console Output
- Overall statistics (total runs, success rate, etc.)
- Category failure breakdown
- Most frequently failed jobs (Top 50) with direct CI links
- Failure pattern analysis

### JSON Export
Detailed analysis data including:
- Complete failure statistics
- Job failure counts
- Workflow failure counts
- Failure patterns
- Recent failure details

## Example Output

```
============================================================
SGLang CI Analysis Report
============================================================

Overall Statistics:
  Total runs: 1000
  Successful: 383
  Failed: 210
  Cancelled: 157
  Skipped: 238
  Success rate: 38.3%

Category Failure Statistics:
  unit-test: 167 failures
  accuracy: 27 failures
  performance: 18 failures
  deepep: 5 failures
  sgl-kernel: 4 failures

Most Frequently Failed Jobs (Top 50):
   1. per-commit-4-ascend-npu: 27 times
      - Run #10066 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053422
      - Run #10061 (2025-09-24 11:04): https://github.com/sgl-project/sglang/actions/runs/17974703166
      - Run #10040 (2025-09-24 08:23): https://github.com/sgl-project/sglang/actions/runs/17970858501
   2. per-commit-2-ascend-npu: 26 times
      - Run #10066 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053422
      - Run #10051 (2025-09-24 09:29): https://github.com/sgl-project/sglang/actions/runs/17972481302
      - Run #10049 (2025-09-24 09:10): https://github.com/sgl-project/sglang/actions/runs/17972001115
   3. per-commit-1-ascend-npu: 25 times
      - Run #10051 (2025-09-24 09:29): https://github.com/sgl-project/sglang/actions/runs/17972481302
      - Run #10049 (2025-09-24 09:10): https://github.com/sgl-project/sglang/actions/runs/17972001115
      - Run #10040 (2025-09-24 08:23): https://github.com/sgl-project/sglang/actions/runs/17970858501
   4. unit-test-backend-1-gpu-amd-mi35x (linux-mi35x-gpu-1): 18 times
      - Run #28891 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053408
      - Run #28890 (2025-09-24 12:20): https://github.com/sgl-project/sglang/actions/runs/17976435645
      - Run #28885 (2025-09-24 11:04): https://github.com/sgl-project/sglang/actions/runs/17974703173
   5. unit-test-backend-1-gpu (5): 15 times
      - Run #34538 (2025-09-24 13:54): https://github.com/sgl-project/sglang/actions/runs/17978943308
      - Run #34536 (2025-09-24 13:35): https://github.com/sgl-project/sglang/actions/runs/17978451365
      - Run #34526 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174403
   6. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 3): 14 times
      - Run #28893 (2025-09-24 13:35): https://github.com/sgl-project/sglang/actions/runs/17978451434
      - Run #28891 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053408
      - Run #28890 (2025-09-24 12:20): https://github.com/sgl-project/sglang/actions/runs/17976435645
   7. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 0): 14 times
      - Run #28891 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053408
      - Run #28890 (2025-09-24 12:20): https://github.com/sgl-project/sglang/actions/runs/17976435645
      - Run #28885 (2025-09-24 11:04): https://github.com/sgl-project/sglang/actions/runs/17974703173
   8. accuracy-test-2-gpu-amd (linux-mi35x-gpu-2): 14 times
      - Run #28883 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174456
      - Run #28879 (2025-09-24 10:26): https://github.com/sgl-project/sglang/actions/runs/17973839120
      - Run #28872 (2025-09-24 09:10): https://github.com/sgl-project/sglang/actions/runs/17972001138
   9. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 5): 11 times
      - Run #28893 (2025-09-24 13:35): https://github.com/sgl-project/sglang/actions/runs/17978451434
      - Run #28891 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053408
      - Run #28890 (2025-09-24 12:20): https://github.com/sgl-project/sglang/actions/runs/17976435645
  10. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 7): 10 times
      - Run #28893 (2025-09-24 13:35): https://github.com/sgl-project/sglang/actions/runs/17978451434
      - Run #28874 (2025-09-24 09:29): https://github.com/sgl-project/sglang/actions/runs/17972481308
      - Run #28872 (2025-09-24 09:10): https://github.com/sgl-project/sglang/actions/runs/17972001138
  11. bench-test-2-gpu-amd (linux-mi300-gpu-2): 9 times
      - Run #28891 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053408
      - Run #28890 (2025-09-24 12:20): https://github.com/sgl-project/sglang/actions/runs/17976435645
      - Run #28879 (2025-09-24 10:26): https://github.com/sgl-project/sglang/actions/runs/17973839120
  12. unit-test-backend-1-gpu (4): 8 times
      - Run #34534 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053442
      - Run #34526 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174403
      - Run #34498 (2025-09-24 07:59): https://github.com/sgl-project/sglang/actions/runs/17970287163
  13. unit-test-backend-2-gpu (0): 7 times
      - Run #34536 (2025-09-24 13:35): https://github.com/sgl-project/sglang/actions/runs/17978451365
      - Run #34526 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174403
      - Run #34520 (2025-09-24 10:09): https://github.com/sgl-project/sglang/actions/runs/17973442270
  14. unit-test-backend-1-gpu (9): 6 times
      - Run #34538 (2025-09-24 13:54): https://github.com/sgl-project/sglang/actions/runs/17978943308
      - Run #34536 (2025-09-24 13:35): https://github.com/sgl-project/sglang/actions/runs/17978451365
      - Run #34526 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174403
  15. unit-test-backend-4-gpu (1): 6 times
      - Run #34502 (2025-09-24 08:23): https://github.com/sgl-project/sglang/actions/runs/17970858519
      - Run #34483 (2025-09-24 06:48): https://github.com/sgl-project/sglang/actions/runs/17968745498
      - Run #34478 (2025-09-24 06:36): https://github.com/sgl-project/sglang/actions/runs/17968517871
  16. unit-test-backend-4-gpu-b200: 6 times
      - Run #34462 (2025-09-24 05:11): https://github.com/sgl-project/sglang/actions/runs/17966982224
      - Run #34454 (2025-09-24 04:26): https://github.com/sgl-project/sglang/actions/runs/17966297243
      - Run #34439 (2025-09-24 03:14): https://github.com/sgl-project/sglang/actions/runs/17965208392
  17. performance-test-1-gpu-part-2-amd (linux-mi300-gpu-1): 5 times
      - Run #28893 (2025-09-24 13:35): https://github.com/sgl-project/sglang/actions/runs/17978451434
      - Run #28883 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174456
      - Run #28879 (2025-09-24 10:26): https://github.com/sgl-project/sglang/actions/runs/17973839120
  18. unit-test-backend-8-gpu-amd (linux-mi300-gpu-8): 5 times
      - Run #28885 (2025-09-24 11:04): https://github.com/sgl-project/sglang/actions/runs/17974703173
      - Run #28874 (2025-09-24 09:29): https://github.com/sgl-project/sglang/actions/runs/17972481308
      - Run #28872 (2025-09-24 09:10): https://github.com/sgl-project/sglang/actions/runs/17972001138
  19. unit-test-backend-2-gpu (1): 5 times
      - Run #34520 (2025-09-24 10:09): https://github.com/sgl-project/sglang/actions/runs/17973442270
      - Run #34473 (2025-09-24 06:15): https://github.com/sgl-project/sglang/actions/runs/17968102842
      - Run #34460 (2025-09-24 05:07): https://github.com/sgl-project/sglang/actions/runs/17966918287
  20. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 6): 5 times
      - Run #28863 (2025-09-24 08:23): https://github.com/sgl-project/sglang/actions/runs/17970858439
      - Run #28862 (2025-09-24 08:23): https://github.com/sgl-project/sglang/actions/runs/17970850775
      - Run #28855 (2025-09-24 07:30): https://github.com/sgl-project/sglang/actions/runs/17969671345
  21. unit-test-deepep-4-gpu: 4 times
      - Run #34527 (2025-09-24 10:43): https://github.com/sgl-project/sglang/actions/runs/17974237736
      - Run #34491 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342167
      - Run #34490 (2025-09-24 07:12): https://github.com/sgl-project/sglang/actions/runs/17969251353
  22. accuracy-test-1-gpu-amd (linux-mi325-gpu-1): 4 times
      - Run #28883 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174456
      - Run #28860 (2025-09-24 07:59): https://github.com/sgl-project/sglang/actions/runs/17970287151
      - Run #28855 (2025-09-24 07:30): https://github.com/sgl-project/sglang/actions/runs/17969671345
  23. performance-test-1-gpu-part-1: 4 times
      - Run #34520 (2025-09-24 10:09): https://github.com/sgl-project/sglang/actions/runs/17973442270
      - Run #34491 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342167
      - Run #34473 (2025-09-24 06:15): https://github.com/sgl-project/sglang/actions/runs/17968102842
  24. performance-test-1-gpu-part-2: 4 times
      - Run #34520 (2025-09-24 10:09): https://github.com/sgl-project/sglang/actions/runs/17973442270
      - Run #34491 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342167
      - Run #34473 (2025-09-24 06:15): https://github.com/sgl-project/sglang/actions/runs/17968102842
  25. unit-test-backend-2-gpu-amd (linux-mi300-gpu-2): 3 times
      - Run #28891 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053408
      - Run #28874 (2025-09-24 09:29): https://github.com/sgl-project/sglang/actions/runs/17972481308
      - Run #28797 (2025-09-24 02:22): https://github.com/sgl-project/sglang/actions/runs/17964402266
  26. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 1): 3 times
      - Run #28885 (2025-09-24 11:04): https://github.com/sgl-project/sglang/actions/runs/17974703173
      - Run #28844 (2025-09-24 06:47): https://github.com/sgl-project/sglang/actions/runs/17968741096
      - Run #28837 (2025-09-24 06:20): https://github.com/sgl-project/sglang/actions/runs/17968205489
  27. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 7): 3 times
      - Run #28885 (2025-09-24 11:04): https://github.com/sgl-project/sglang/actions/runs/17974703173
      - Run #28879 (2025-09-24 10:26): https://github.com/sgl-project/sglang/actions/runs/17973839120
      - Run #28839 (2025-09-24 06:30): https://github.com/sgl-project/sglang/actions/runs/17968407818
  28. unit-test-backend-1-gpu (8): 3 times
      - Run #34526 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174403
      - Run #34522 (2025-09-24 10:26): https://github.com/sgl-project/sglang/actions/runs/17973839111
      - Run #34421 (2025-09-24 02:19): https://github.com/sgl-project/sglang/actions/runs/17964353674
  29. unit-test-frontend: 3 times
      - Run #34520 (2025-09-24 10:09): https://github.com/sgl-project/sglang/actions/runs/17973442270
      - Run #34473 (2025-09-24 06:15): https://github.com/sgl-project/sglang/actions/runs/17968102842
      - Run #34423 (2025-09-24 02:20): https://github.com/sgl-project/sglang/actions/runs/17964379996
  30. accuracy-test-1-gpu: 3 times
      - Run #34520 (2025-09-24 10:09): https://github.com/sgl-project/sglang/actions/runs/17973442270
      - Run #34473 (2025-09-24 06:15): https://github.com/sgl-project/sglang/actions/runs/17968102842
      - Run #34423 (2025-09-24 02:20): https://github.com/sgl-project/sglang/actions/runs/17964379996
  31. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 0): 3 times
      - Run #28860 (2025-09-24 07:59): https://github.com/sgl-project/sglang/actions/runs/17970287151
      - Run #28852 (2025-09-24 07:12): https://github.com/sgl-project/sglang/actions/runs/17969251355
      - Run #28850 (2025-09-24 07:01): https://github.com/sgl-project/sglang/actions/runs/17969003175
  32. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 3): 3 times
      - Run #28853 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342193
      - Run #28830 (2025-09-24 05:35): https://github.com/sgl-project/sglang/actions/runs/17967409723
      - Run #28794 (2025-09-24 02:20): https://github.com/sgl-project/sglang/actions/runs/17964379983
  33. accuracy-test-2-gpu-amd (linux-mi300-gpu-2): 3 times
      - Run #28847 (2025-09-24 06:57): https://github.com/sgl-project/sglang/actions/runs/17968932516
      - Run #28840 (2025-09-24 06:36): https://github.com/sgl-project/sglang/actions/runs/17968517855
      - Run #28830 (2025-09-24 05:35): https://github.com/sgl-project/sglang/actions/runs/17967409723
  34. mla-test-1-gpu-amd (linux-mi325-gpu-1): 3 times
      - Run #28839 (2025-09-24 06:30): https://github.com/sgl-project/sglang/actions/runs/17968407818
      - Run #28797 (2025-09-24 02:22): https://github.com/sgl-project/sglang/actions/runs/17964402266
      - Run #28794 (2025-09-24 02:20): https://github.com/sgl-project/sglang/actions/runs/17964379983
  35. build-test (all): 2 times
      - Run #15749 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053404
      - Run #15652 (2025-09-24 02:22): https://github.com/sgl-project/sglang/actions/runs/17964402263
  36. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 5): 2 times
      - Run #28890 (2025-09-24 12:20): https://github.com/sgl-project/sglang/actions/runs/17976435645
      - Run #28855 (2025-09-24 07:30): https://github.com/sgl-project/sglang/actions/runs/17969671345
  37. unit-test-backend-8-gpu (0): 2 times
      - Run #34528 (2025-09-24 11:04): https://github.com/sgl-project/sglang/actions/runs/17974703178
      - Run #34512 (2025-09-24 09:10): https://github.com/sgl-project/sglang/actions/runs/17972001211
  38. bench-test-2-gpu-amd (linux-mi325-gpu-2): 2 times
      - Run #28883 (2025-09-24 10:40): https://github.com/sgl-project/sglang/actions/runs/17974174456
      - Run #28853 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342193
  39. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 4): 2 times
      - Run #28874 (2025-09-24 09:29): https://github.com/sgl-project/sglang/actions/runs/17972481308
      - Run #28853 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342193
  40. per-commit-16-ascend-a3: 2 times
      - Run #10030 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342205
      - Run #9975 (2025-09-24 02:22): https://github.com/sgl-project/sglang/actions/runs/17964402265
  41. performance-test-1-gpu-part-1-amd (linux-mi325-gpu-1): 2 times
      - Run #28853 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342193
      - Run #28850 (2025-09-24 07:01): https://github.com/sgl-project/sglang/actions/runs/17969003175
  42. unit-test-backend-1-gpu (2): 2 times
      - Run #34491 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342167
      - Run #34454 (2025-09-24 04:26): https://github.com/sgl-project/sglang/actions/runs/17966297243
  43. performance-test-2-gpu: 2 times
      - Run #34491 (2025-09-24 07:16): https://github.com/sgl-project/sglang/actions/runs/17969342167
      - Run #34475 (2025-09-24 06:20): https://github.com/sgl-project/sglang/actions/runs/17968205483
  44. unit-test-backend-1-gpu (3): 2 times
      - Run #34490 (2025-09-24 07:12): https://github.com/sgl-project/sglang/actions/runs/17969251353
      - Run #34460 (2025-09-24 05:07): https://github.com/sgl-project/sglang/actions/runs/17966918287
  45. accuracy-test-1-gpu-amd (linux-mi300-gpu-1): 2 times
      - Run #28844 (2025-09-24 06:47): https://github.com/sgl-project/sglang/actions/runs/17968741096
      - Run #28797 (2025-09-24 02:22): https://github.com/sgl-project/sglang/actions/runs/17964402266
  46. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 4): 2 times
      - Run #28840 (2025-09-24 06:36): https://github.com/sgl-project/sglang/actions/runs/17968517855
      - Run #28839 (2025-09-24 06:30): https://github.com/sgl-project/sglang/actions/runs/17968407818
  47. mla-test-1-gpu-amd (linux-mi300-gpu-1): 2 times
      - Run #28837 (2025-09-24 06:20): https://github.com/sgl-project/sglang/actions/runs/17968205489
      - Run #28794 (2025-09-24 02:20): https://github.com/sgl-project/sglang/actions/runs/17964379983
  48. sgl-kernel-mla-test: 2 times
      - Run #34473 (2025-09-24 06:15): https://github.com/sgl-project/sglang/actions/runs/17968102842
      - Run #34423 (2025-09-24 02:20): https://github.com/sgl-project/sglang/actions/runs/17964379996
  49. sgl-kernel-unit-test: 2 times
      - Run #34473 (2025-09-24 06:15): https://github.com/sgl-project/sglang/actions/runs/17968102842
      - Run #34423 (2025-09-24 02:20): https://github.com/sgl-project/sglang/actions/runs/17964379996
  50. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 2): 1 times
      - Run #28891 (2025-09-24 12:44): https://github.com/sgl-project/sglang/actions/runs/17977053408

Failure Pattern Analysis:
  Unit Test Failure: 164 times
  Other: 82 times
  Accuracy Test Failure: 27 times
  Performance Test Failure: 18 times
  Dependency Installation Failure: 17 times
  GPU Related Failure: 16 times
  Build Failure: 1 times
```

## CI Job Categories

The tool automatically categorizes CI jobs into:

- **sgl-kernel**: Kernel-related tests (build, unit tests, MLA tests)
- **unit-test**: Unit tests (frontend, backend with different GPU counts)
- **performance**: Performance tests (latency, throughput benchmarks)
- **accuracy**: Accuracy tests (model evaluation)
- **deepep**: DeepEP-related tests
- **b200**: B200 hardware-specific tests

## Failure Patterns

The tool recognizes these failure patterns:

- **Timeout**: Step execution timeout
- **Unit Test Failure**: Unit test execution failures
- **Performance Test Failure**: Performance benchmark failures
- **Accuracy Test Failure**: Model accuracy evaluation failures
- **Build Failure**: Compilation/build process failures
- **Dependency Installation Failure**: Package installation issues
- **GPU Related Failure**: GPU-specific test failures
- **Other**: Unclassified failures

## Troubleshooting

### Common Issues

1. **404 Error**:
   - Ensure the repository name is correct (`sgl-project/sglang`)
   - **Most common cause**: Missing `repo` or `workflow` permissions in your GitHub token
   - Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens) and regenerate with correct permissions
2. **403 Error**: Check that your GitHub token has the correct permissions (`repo` and `workflow`)
3. **Rate Limiting**: The tool includes built-in delays to avoid API rate limits
4. **Network Issues**: Ensure stable internet connection

### Debug Mode

For detailed API call information, you can modify the code to include logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This tool follows the same license as the SGLang project.
