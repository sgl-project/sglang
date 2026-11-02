"use strict";

/**
 * Comments on a PR when a required metadata check (PR title, lint) fails.
 *
 * Runs from workflow_run, which gets a write token even for fork PRs. It reads
 * only API metadata, never PR code or artifacts. A new comment is posted only
 * when a failed run has not been reported yet; older comments are minimized.
 */

const MARKER = "<!-- pr-check-comment";
const BOT = "github-actions[bot]";

const CHECKS = [
  {
    name: "PR title",
    workflow: "pr-title.yml",
    event: "pull_request_target",
    // pull_request_target runs main's workflow, so its error text is ours.
    quoteErrors: true,
    hint: "Edit the title; the check reruns on its own.",
  },
  {
    name: "Lint",
    workflow: "lint.yml",
    event: "pull_request",
    // A PR can change what its own pull_request run prints; never quote it.
    quoteErrors: false,
    hint:
      "Run `pre-commit run --all-files` locally and push the result. " +
      "The run log shows any other failing step.",
  },
];

// workflow_run.pull_requests is empty for fork PRs; fall back to the head ref.
async function findPr(github, context, wr) {
  const repo = context.repo;
  let number = wr.pull_requests?.[0]?.number;
  if (!number && wr.head_repository && wr.head_branch) {
    const { data: prs } = await github.rest.pulls.list({
      ...repo,
      state: "open",
      head: `${wr.head_repository.owner.login}:${wr.head_branch}`,
    });
    number = (prs.find((p) => p.head.sha === wr.head_sha) || prs[0])?.number;
  }
  if (!number) return null;
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: number });
  return pr;
}

async function latestRun(github, context, check, sha) {
  const { data } = await github.rest.actions.listWorkflowRuns({
    ...context.repo,
    workflow_id: check.workflow,
    head_sha: sha,
    event: check.event,
    per_page: 10,
  });
  return data.workflow_runs[0] || null;
}

async function errorMessages(github, context, run) {
  const { data } = await github.rest.actions.listJobsForWorkflowRun({
    ...context.repo,
    run_id: run.id,
  });
  const messages = [];
  for (const job of data.jobs.filter((j) => j.conclusion === "failure")) {
    const { data: annotations } = await github.rest.checks.listAnnotations({
      ...context.repo,
      check_run_id: job.id,
    });
    for (const a of annotations) {
      if (a.annotation_level === "failure" && !/^Process completed with exit code/.test(a.message)) {
        messages.push(a.message);
      }
    }
  }
  return messages;
}

function reportedRunIds(body) {
  const match = /^<!-- pr-check-comment runs=([\d,]+) -->/.exec(body);
  return match ? match[1].split(",").map(Number) : [];
}

async function buildBody(github, context, sha, failed) {
  const lines = [
    `${MARKER} runs=${failed.map(({ run }) => run.id).join(",")} -->`,
    `Required checks failed on ${sha.slice(0, 7)}; this PR cannot be merged until they pass.`,
  ];
  for (const { check, run } of failed) {
    lines.push("", `**${check.name}** ([run](${run.html_url}))`);
    if (check.quoteErrors) {
      for (const message of await errorMessages(github, context, run)) {
        lines.push(`> ${message.replace(/\n/g, "\n> ")}`);
      }
    }
    lines.push(check.hint);
  }
  return lines.join("\n");
}

async function minimizeComments(github, comments) {
  if (comments.length === 0) return;
  const { nodes } = await github.graphql(
    "query($ids: [ID!]!) { nodes(ids: $ids) { ... on IssueComment { id isMinimized } } }",
    { ids: comments.map((c) => c.node_id) }
  );
  for (const node of nodes.filter((n) => n && !n.isMinimized)) {
    await github.graphql(
      "mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) { clientMutationId } }",
      { id: node.id }
    );
  }
}

async function run({ github, context, core }) {
  const wr = context.payload.workflow_run;
  if (!CHECKS.some((check) => check.event === wr.event)) {
    core.info(`Triggering run event '${wr.event}' is not a PR event; skipping.`);
    return;
  }
  const pr = await findPr(github, context, wr);
  if (!pr || pr.state !== "open" || pr.draft) {
    core.info("No open, ready PR for this run; skipping.");
    return;
  }
  if (pr.head.sha !== wr.head_sha) {
    core.info(`Run is for ${wr.head_sha}, PR head is ${pr.head.sha}; skipping.`);
    return;
  }

  const runs = await Promise.all(
    CHECKS.map((check) => latestRun(github, context, check, pr.head.sha))
  );
  // The check still running triggers this workflow again when it completes.
  if (runs.some((r) => r && r.status !== "completed")) {
    core.info("A check is still running; waiting for it.");
    return;
  }
  const failed = CHECKS.map((check, i) => ({ check, run: runs[i] })).filter(
    ({ run }) => run?.conclusion === "failure"
  );

  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo,
    issue_number: pr.number,
    per_page: 100,
  });
  const previous = comments.filter((c) => c.user?.login === BOT && c.body?.startsWith(MARKER));

  if (failed.length > 0) {
    const reported = new Set(previous.flatMap((c) => reportedRunIds(c.body)));
    if (failed.every(({ run }) => reported.has(run.id))) {
      core.info("These failures are already reported.");
      return;
    }
    await github.rest.issues.createComment({
      ...context.repo,
      issue_number: pr.number,
      body: await buildBody(github, context, pr.head.sha, failed),
    });
  } else {
    core.info("All checks pass.");
  }
  await minimizeComments(github, previous);
}

module.exports = { run };
