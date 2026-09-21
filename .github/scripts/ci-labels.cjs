"use strict";

/**
 * Resolve a PR's CI control labels into the axes the pipeline dispatches on.
 *
 * Four labels, three of them independent switches and one that turns on all
 * three:
 *
 *   bypass-fail-fast  a sibling job failing does not abort the rest of the run
 *   parallel-stages   stages dispatch together instead of waiting on each other
 *   max-concurrency   a suite fans out to every shard at once, not 1/3 of them
 *   highest-priority  all three, plus the protections keyed on it elsewhere
 *                     (never batch-cancelled, sorted first, never stale-closed)
 *
 * Labels come from the API, not from `context.payload`: a rerun replays the
 * original event, so the payload still carries the label set from when the run
 * was first created. Adding a label and rerunning is how a PR opts in, and only
 * a live read observes that.
 */

const BYPASS_FAIL_FAST = "bypass-fail-fast";
const PARALLEL_STAGES = "parallel-stages";
const MAX_CONCURRENCY = "max-concurrency";
const HIGHEST_PRIORITY = "highest-priority";

async function readLabels(github, context) {
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
