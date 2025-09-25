# SGLang CI Monitor

A simple tool to analyze CI failures for the SGLang project. This tool fetches recent CI run data from GitHub Actions and provides detailed analysis of failure patterns.

## Features

- **Simple Analysis**: Analyze recent CI runs and identify failure patterns
- **Category Classification**: Automatically categorize failures by type (unit-test, performance, etc.)
- **Pattern Recognition**: Identify common failure patterns (timeouts, build failures, etc.)
- **CI Links**: Direct links to recent failed CI runs for detailed investigation
- **Last Success Tracking**: Track the last successful run for each failed job with PR information
- **JSON Export**: Export detailed analysis data to JSON format
- **Automated Monitoring**: GitHub Actions workflow for continuous CI monitoring

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
  Successful: 392
  Failed: 187
  Cancelled: 181
  Skipped: 150
  Success rate: 39.2%

Category Failure Statistics:
  unit-test: 351 failures
  accuracy: 84 failures
  performance: 55 failures
  deepep: 1 failures

Most Frequently Failed Jobs (Top 50):
   1. unit-test-backend-1-gpu-amd-mi35x (linux-mi35x-gpu-1): 32 times
      Last Success: Run #28893 (2025-09-24 13:35) by Xiaoze Fan: https://github.com/sgl-project/sglang/actions/runs/17978451434
      Recent Failures:
        - Run #28958 (2025-09-25 01:51) (PR #1 by Yuhao Yao): https://github.com/sgl-project/sglang/actions/runs/17994520789
        - Run #28957 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860400
        - Run #28956 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826732
   2. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 3): 31 times
      Last Success: Run #28903 (2025-09-24 15:38) by gholmes829: https://github.com/sgl-project/sglang/actions/runs/17981905113
      Recent Failures:
        - Run #28958 (2025-09-25 01:51) (PR #1 by Yuhao Yao): https://github.com/sgl-project/sglang/actions/runs/17994520789
        - Run #28957 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860400
        - Run #28956 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826732
   3. accuracy-test-2-gpu-amd (linux-mi35x-gpu-2): 29 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28958 (2025-09-25 01:51) (PR #1 by Yuhao Yao): https://github.com/sgl-project/sglang/actions/runs/17994520789
        - Run #28957 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860400
        - Run #28956 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826732
   4. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 5): 23 times
      Last Success: Run #28906 (2025-09-24 15:43) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17982029749
      Recent Failures:
        - Run #28958 (2025-09-25 01:51) (PR #1 by Yuhao Yao): https://github.com/sgl-project/sglang/actions/runs/17994520789
        - Run #28956 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826732
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
   5. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 0): 23 times
      Last Success: Run #28893 (2025-09-24 13:35) by Xiaoze Fan: https://github.com/sgl-project/sglang/actions/runs/17978451434
      Recent Failures:
        - Run #28956 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826732
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
   6. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 7): 18 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28956 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826732
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
   7. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 3): 17 times
      Last Success: Run #28893 (2025-09-24 13:35) by Xiaoze Fan: https://github.com/sgl-project/sglang/actions/runs/17978451434
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
   8. build-test (all): 16 times
      Last Success: Run #15748 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435618
      Recent Failures:
        - Run #15824 (2025-09-25 02:16) by Yuan Luo: https://github.com/sgl-project/sglang/actions/runs/17994892894
        - Run #15814 (2025-09-25 00:53) by diwei sun: https://github.com/sgl-project/sglang/actions/runs/17993616261
        - Run #15812 (2025-09-25 00:35) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993338746
   9. bench-test-2-gpu-amd (linux-mi300-gpu-2): 15 times
      Last Success: Run #28893 (2025-09-24 13:35) by Xiaoze Fan: https://github.com/sgl-project/sglang/actions/runs/17978451434
      Recent Failures:
        - Run #28957 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860400
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  10. performance-test-1-gpu-part-2-amd (linux-mi300-gpu-1): 15 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  11. accuracy-test-1-gpu-amd (linux-mi325-gpu-1): 15 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  12. unit-test-backend-8-gpu-amd (linux-mi300-gpu-8): 15 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  13. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 1): 14 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  14. unit-test-backend-2-gpu-amd (linux-mi300-gpu-2): 14 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  15. performance-test-1-gpu-part-1-amd (linux-mi325-gpu-1): 13 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  16. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 2): 13 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  17. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 4): 13 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  18. accuracy-test-2-gpu-amd (linux-mi325-gpu-2): 13 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  19. mla-test-1-gpu-amd (linux-mi325-gpu-1): 13 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  20. accuracy-test-2-gpu-amd (linux-mi300-gpu-2): 13 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  21. accuracy-test-1-gpu-amd (linux-mi300-gpu-1): 12 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  22. performance-test-1-gpu-part-2-amd (linux-mi325-gpu-1): 12 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  23. bench-test-2-gpu-amd (linux-mi325-gpu-2): 11 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28957 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860400
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  24. unit-test-sgl-kernel-amd (linux-mi325-gpu-1): 11 times
      Last Success: Run #28891 (2025-09-24 12:44) by Yuan Luo: https://github.com/sgl-project/sglang/actions/runs/17977053408
      Recent Failures:
        - Run #28956 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826732
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  25. performance-test-1-gpu-part-1-amd (linux-mi300-gpu-1): 11 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  26. unit-test-backend-1-gpu-amd (linux-mi300-gpu-1, 6): 11 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  27. unit-test-backend-2-gpu-amd (linux-mi325-gpu-2): 11 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  28. unit-test-backend-1-gpu (9): 10 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34623 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826751
        - Run #34617 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619818
        - Run #34581 (2025-09-24 19:49) by Yineng Zhang: https://github.com/sgl-project/sglang/actions/runs/17987860976
  29. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 0): 10 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
  30. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 1): 10 times
      Last Success: Run #28891 (2025-09-24 12:44) by Yuan Luo: https://github.com/sgl-project/sglang/actions/runs/17977053408
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  31. mla-test-1-gpu-amd (linux-mi300-gpu-1): 10 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  32. unit-test-backend-1-gpu (5): 9 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34624 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860412
        - Run #34617 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619818
        - Run #34560 (2025-09-24 17:01) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17983919007
  33. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 2): 9 times
      Last Success: Run #28906 (2025-09-24 15:43) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17982029749
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  34. unit-test-sgl-kernel-amd (linux-mi300-gpu-1): 9 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28952 (2025-09-24 23:57) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992751764
        - Run #28951 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619816
  35. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 4): 7 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28955 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426068
        - Run #28953 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178855
        - Run #28949 (2025-09-24 23:44) (PR #10372 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992591372
  36. unit-test-backend-1-gpu-amd (linux-mi325-gpu-1, 6): 7 times
      Last Success: Run #28890 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435645
      Recent Failures:
        - Run #28950 (2025-09-24 23:45) (PR #1 by Xiaoyu Zhang): https://github.com/sgl-project/sglang/actions/runs/17992598523
        - Run #28946 (2025-09-24 23:39) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992521547
        - Run #28936 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244192
  37. vllm-dependency-test: 6 times
      Last Success: Run #22949 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435651
      Recent Failures:
        - Run #23028 (2025-09-25 02:39) by xuyongfei.xyf: https://github.com/sgl-project/sglang/actions/runs/17995251178
        - Run #23021 (2025-09-25 02:16) by Yuan Luo: https://github.com/sgl-project/sglang/actions/runs/17994892873
        - Run #22993 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244213
  38. per-commit-4-ascend-npu: 6 times
      Last Success: Run #10065 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435703
      Recent Failures:
        - Run #10138 (2025-09-25 02:17) by wangyi: https://github.com/sgl-project/sglang/actions/runs/17994908950
        - Run #10137 (2025-09-25 02:16) by Yuan Luo: https://github.com/sgl-project/sglang/actions/runs/17994892896
        - Run #10124 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619819
  39. unit-test-backend-2-gpu (0): 6 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34624 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860412
        - Run #34593 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244227
        - Run #34576 (2025-09-24 18:46) by eigen: https://github.com/sgl-project/sglang/actions/runs/17986403452
  40. unit-test-backend-1-gpu (4): 6 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34623 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826751
        - Run #34609 (2025-09-24 23:25) (PR #10853 by Yineng Zhang): https://github.com/sgl-project/sglang/actions/runs/17992311361
        - Run #34560 (2025-09-24 17:01) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17983919007
  41. run-all-notebooks: 6 times
      Last Success: Run #26939 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435610
      Recent Failures:
        - Run #26988 (2025-09-24 23:25) (PR #10853 by Yineng Zhang): https://github.com/sgl-project/sglang/actions/runs/17992311396
        - Run #26982 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244193
        - Run #26973 (2025-09-24 18:46) by eigen: https://github.com/sgl-project/sglang/actions/runs/17986403458
  42. per-commit-2-ascend-npu: 5 times
      Last Success: Run #10065 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435703
      Recent Failures:
        - Run #10135 (2025-09-25 02:16) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17994888152
        - Run #10109 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244207
        - Run #10085 (2025-09-24 16:42) by likesen: https://github.com/sgl-project/sglang/actions/runs/17983486537
  43. unit-test-backend-8-gpu (0): 5 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34623 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826751
        - Run #34621 (2025-09-25 00:40) by Hubert Lu: https://github.com/sgl-project/sglang/actions/runs/17993426098
        - Run #34619 (2025-09-25 00:24) (PR #10372 by BBuf): https://github.com/sgl-project/sglang/actions/runs/17993178853
  44. pytest-rust: 5 times
      Last Success: Run #1761 (2025-09-24 16:39) by Chang Su: https://github.com/sgl-project/sglang/actions/runs/17983415401
      Recent Failures:
        - Run #1770 (2025-09-24 21:02) by Simo Lin: https://github.com/sgl-project/sglang/actions/runs/17989538977
        - Run #1769 (2025-09-24 20:54) by Simo Lin: https://github.com/sgl-project/sglang/actions/runs/17989380799
        - Run #1767 (2025-09-24 20:36) by Ata Fatahi: https://github.com/sgl-project/sglang/actions/runs/17988964074
  45. per-commit-16-ascend-a3: 4 times
      Last Success: Run #10065 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435703
      Recent Failures:
        - Run #10138 (2025-09-25 02:17) by wangyi: https://github.com/sgl-project/sglang/actions/runs/17994908950
        - Run #10135 (2025-09-25 02:16) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17994888152
        - Run #10109 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244207
  46. unit-test-backend-1-gpu (7): 4 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34624 (2025-09-25 01:10) (PR #10883 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993860412
        - Run #34573 (2025-09-24 18:45) by Tejesh Anand: https://github.com/sgl-project/sglang/actions/runs/17986382981
        - Run #34565 (2025-09-24 17:35) by YAMY: https://github.com/sgl-project/sglang/actions/runs/17984740528
  47. unit-test-backend-2-gpu (1): 4 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34593 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244227
        - Run #34576 (2025-09-24 18:46) by eigen: https://github.com/sgl-project/sglang/actions/runs/17986403452
        - Run #34565 (2025-09-24 17:35) by YAMY: https://github.com/sgl-project/sglang/actions/runs/17984740528
  48. per-commit-1-ascend-npu: 3 times
      Last Success: Run #10065 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435703
      Recent Failures:
        - Run #10138 (2025-09-25 02:17) by wangyi: https://github.com/sgl-project/sglang/actions/runs/17994908950
        - Run #10109 (2025-09-24 21:32) by xiafang: https://github.com/sgl-project/sglang/actions/runs/17990244207
        - Run #10085 (2025-09-24 16:42) by likesen: https://github.com/sgl-project/sglang/actions/runs/17983486537
  49. unit-test-backend-1-gpu (1): 3 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34623 (2025-09-25 01:07) (PR #10495 by Lianmin Zheng): https://github.com/sgl-project/sglang/actions/runs/17993826751
        - Run #34554 (2025-09-24 16:29) by Yuan Luo: https://github.com/sgl-project/sglang/actions/runs/17983177051
        - Run #34548 (2025-09-24 15:38) by gholmes829: https://github.com/sgl-project/sglang/actions/runs/17981905143
  50. unit-test-backend-1-gpu (8): 3 times
      Last Success: Run #34533 (2025-09-24 12:20) by Yuhong Guo: https://github.com/sgl-project/sglang/actions/runs/17976435636
      Recent Failures:
        - Run #34617 (2025-09-24 23:47) (PR #10881 by Chang Su): https://github.com/sgl-project/sglang/actions/runs/17992619818
        - Run #34581 (2025-09-24 19:49) by Yineng Zhang: https://github.com/sgl-project/sglang/actions/runs/17987860976
        - Run #34554 (2025-09-24 16:29) by Yuan Luo: https://github.com/sgl-project/sglang/actions/runs/17983177051

Failure Pattern Analysis:
  GPU Related Failure: 223 times
  Unit Test Failure: 190 times
  Accuracy Test Failure: 84 times
  Performance Test Failure: 54 times
  Other: 34 times
  Dependency Installation Failure: 19 times
  Build Failure: 15 times
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

## Automated Monitoring

The CI monitor is also available as a GitHub Actions workflow that runs automatically every 6 hours. The workflow:

- Analyzes the last 500 CI runs
- Generates detailed reports
- Uploads analysis results as artifacts

### Workflow Configuration

The workflow is located at `.github/workflows/ci-monitor.yml` and uses the `GH_PAT_FOR_NIGHTLY_CI` secret for GitHub API access.

### Manual Trigger

You can manually trigger the workflow from the GitHub Actions tab with custom parameters:
- `limit`: Number of CI runs to analyze (default: 500)

## License

This tool follows the same license as the SGLang project.
