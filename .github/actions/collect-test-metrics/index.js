const core = require('@actions/core');
const exec = require('@actions/exec');
const { DefaultArtifactClient } = require('@actions/artifact');
const fs = require('fs');
const path = require('path');

async function main() {
  try {
    // Get inputs
    const metricsDirInput = core.getInput('metrics_dir');
    const outputFile = core.getInput('output_file');
    const artifactName = core.getInput('artifact_name');

    // Handle absolute vs relative path
    const metricsDir = path.isAbsolute(metricsDirInput)
      ? metricsDirInput
      : path.join(process.env.GITHUB_WORKSPACE, metricsDirInput);

    core.info(`Setting up metrics collection in: ${metricsDir}`);

    // Clean up and create metrics directory (equivalent to rm -rf && mkdir -p)
    if (fs.existsSync(metricsDir)) {
      core.info(`Cleaning existing metrics directory`);
      fs.rmSync(metricsDir, { recursive: true, force: true });
    }

    fs.mkdirSync(metricsDir, { recursive: true });
    core.info(`Created metrics directory: ${metricsDir}`);

    // Set environment variable for tests to match workflow global env pattern
    // SGLANG_TEST_METRICS_OUTPUT = ${metricsDir}/test_metrics
    const metricsBasePath = path.join(metricsDir, 'test_metrics');
    core.exportVariable('SGLANG_TEST_METRICS_OUTPUT', metricsBasePath);
    core.info(`Exported SGLANG_TEST_METRICS_OUTPUT=${metricsBasePath}`);

    // Save state for post action
    core.saveState('metrics_dir', metricsDir);
    core.saveState('output_file', outputFile);
    core.saveState('artifact_name', artifactName);
    core.saveState('metrics_base_path', metricsBasePath);

    core.info('Metrics collection setup complete');
  } catch (error) {
    core.setFailed(`Main action failed: ${error.message}`);
  }
}

async function post() {
  try {
    // Retrieve saved state
    const metricsDir = core.getState('metrics_dir');
    const outputFile = core.getState('output_file');
    const artifactName = core.getState('artifact_name');
    const metricsBasePath = core.getState('metrics_base_path');

    if (!metricsDir) {
      core.warning('No metrics_dir state found, skipping post action');
      return;
    }

    core.info('Starting metrics post-processing');

    // Check if metrics directory exists
    if (!fs.existsSync(metricsDir)) {
      core.warning(`Metrics directory not found: ${metricsDir}`);
      return;
    }

    // Run merge_metrics.py
    const outputFilePath = path.join(metricsDir, outputFile);
    const mergeScriptPath = path.join(process.env.GITHUB_WORKSPACE, 'scripts/ci/merge_metrics.py');

    core.info(`Merging metrics: ${metricsBasePath} -> ${outputFilePath}`);

    try {
      await exec.exec('python3', [mergeScriptPath, metricsBasePath, outputFilePath]);
      core.info('Metrics merged successfully');
    } catch (error) {
      core.warning(`Failed to merge metrics: ${error.message}`);
      // Continue to upload even if merge fails
    }

    // Upload artifacts using @actions/artifact
    core.info(`Uploading artifact: ${artifactName}`);

    try {
      const artifactClient = new DefaultArtifactClient();

      // Get all files in metrics directory (skip subdirectories)
      const files = fs.readdirSync(metricsDir)
        .map(file => path.join(metricsDir, file))
        .filter(fullPath => {
          try {
            return fs.statSync(fullPath).isFile();
          } catch (err) {
            core.warning(`Could not stat ${fullPath}: ${err.message}`);
            return false;
          }
        });

      if (files.length === 0) {
        core.warning('No metrics files found to upload');
        return;
      }

      core.info(`Found ${files.length} file(s) to upload`);

      // Upload artifact
      const uploadResponse = await artifactClient.uploadArtifact(
        artifactName,
        files,
        metricsDir,
        {
          continueOnError: true
        }
      );

      core.info(`Artifact uploaded successfully: ${uploadResponse.artifactName}`);
    } catch (error) {
      core.warning(`Failed to upload artifact: ${error.message}`);
      // Don't fail the job - metrics upload is supplementary
    }

  } catch (error) {
    // Post actions must not fail the job
    core.warning(`Post action error: ${error.message}`);
  }
}

// Determine if this is main or post execution
// GitHub Actions sets STATE_* env vars for post hooks
const isPost = !!process.env['STATE_metrics_dir'];

if (isPost) {
  post().catch(err => {
    core.warning(`Post hook failed: ${err.message}`);
  });
} else {
  main().catch(err => {
    core.setFailed(`Main hook failed: ${err.message}`);
  });
}
