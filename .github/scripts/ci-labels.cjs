"use strict";

/**
 * Labels come from the API, not `context.payload`: a rerun replays the original
 * event, so the payload carries the label set from when the run was created.
 */

const BYPASS_FAIL_FAST = "bypass-fail-fast";
const PARALLEL_STAGES = "parallel-stages";
const MAX_CONCURRENCY = "max-concurrency";
const HIGHEST_PRIORITY = "highest-priority";

// Callers gate the whole run on this, so a transient API error must not fail them.
async function readLabels(github, context) {
  try {
    return await fetchLabels(github, context);
  } catch (e) {
    console.warn(`Could not read PR labels: ${e.message}`);
    return [];
  }
}

async function fetchLabels(github, context) {
  const prNumber = context.payload.pull_request?.number;
  if (prNumber) {
    const { data: pr } = await github.rest.pulls.get({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: prNumber,
    });
    return pr.labels.map((l) => l.name);
  }
  const sha = context.payload.pull_request?.head?.sha || context.sha;
  const { data: prs } =
    await github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner: context.repo.owner,
      repo: context.repo.repo,
      commit_sha: sha,
    });
  return prs.length > 0 ? prs[0].labels.map((l) => l.name) : [];
}

async function resolveCiLabels(github, context) {
  const labels = await readLabels(github, context);
  const has = (name) => labels.includes(name);
  const highestPriority = has(HIGHEST_PRIORITY);
  return {
    labels,
    bypassFailFast: highestPriority || has(BYPASS_FAIL_FAST),
    parallelStages: highestPriority || has(PARALLEL_STAGES),
    maxConcurrency: highestPriority || has(MAX_CONCURRENCY),
    highestPriority,
  };
}

module.exports = {
  resolveCiLabels,
  BYPASS_FAIL_FAST,
  PARALLEL_STAGES,
  MAX_CONCURRENCY,
  HIGHEST_PRIORITY,
};
