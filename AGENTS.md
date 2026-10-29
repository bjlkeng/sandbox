# Repository instructions

## Project context

This repository is a collection of experiments and notebooks supporting blog
posts. Its purpose is learning: favor small, readable implementations, clear
connections to the underlying math, and reproducible results. Read the root
README and the relevant experiment directory's README and design notes before
implementing a bead. Keep dependencies and code scoped to that experiment.
For the reinforcement learning work, consult `rl_math/PLAN.md` when present.

The user reviews each implementation PR and merges it manually. Implementation,
passing checks, and a merged PR do not by themselves authorize closing a bead.
This file is the workflow policy. Its branch, review, and closure rules take
precedence over optional Beads skills and default `bd prime` guidance.

## Separate Beads workspace for each subproject

Each subproject owns its own Beads database, configuration, issue prefix,
dependencies, memories, and viewer export. For example:

```text
sandbox/
  AGENTS.md
  .beads/                 # Repository-wide maintenance only
  rl_math/
    .beads/               # RL issues; prefix rlmath
  hmc/
    .beads/               # HMC issues; prefix hmc
```

Create a workspace when work on that subproject first needs tracking. Keep RL
tasks in `rl_math/.beads/`, HMC tasks in `hmc/.beads/`, and other tasks in their
own subprojects. Root `.beads/` is reserved for repository-wide work such as
these instructions. Separate epics or labels in one shared database do not
provide the requested separation.

Select the subproject before reading, creating, claiming, or updating issues.
Beads normally discovers a shared repository workspace from linked worktrees;
use an explicit absolute `BEADS_DIR` for every command to prevent routing to the
root tracker, another project, or a fresh database in the feature worktree.
All worktrees working on the same subproject reuse that subproject's database
in the primary checkout. Different subprojects use different databases.

For example, from any checkout of this non-bare Git repository:

```bash
sandbox_common=$(git rev-parse --path-format=absolute --git-common-dir)
sandbox_primary=$(dirname "$sandbox_common")
sandbox_beads="$sandbox_primary/rl_math/.beads"
BEADS_DIR="$sandbox_beads" bd where
BEADS_DIR="$sandbox_beads" bd prime
BEADS_DIR="$sandbox_beads" bd ready
```

Set `sandbox_beads` to the selected project's path, or to
`$sandbox_primary/.beads` for repository maintenance. Confirm `bd where` reports
the intended path and prefix before making issue changes. Keep the binding
explicit on each command instead of exporting one project globally across
unrelated work. Pass it to Git commands that invoke Beads hooks as well, for
example `BEADS_DIR="$sandbox_beads" git commit ...`.

When initializing a new project, run from its directory in the primary checkout
with its explicit `BEADS_DIR` and a distinct prefix. Use
`bd init --prefix <project-prefix> --non-interactive --skip-agents --skip-hooks --remote ''`
with that binding. The explicit empty remote keeps independent databases from
automatically targeting the same Git-backed Dolt history. Initialization may
create a scoped local Git commit; inspect it before publishing the bead's PR.
Add a short `<project>/.beads/PRIME.md` using the same policy pointer as the root
`.beads/PRIME.md`. Keep database and runtime files ignored by Git.

Do not install per-project hooks that replace repository-wide hooks, copy a
database into a feature worktree, or initialize new databases in every linked
worktree. Separate remote synchronization, if requested, must use isolated
project destinations; distinct issue prefixes alone do not isolate remote data.

## Viewing a subproject with `bv`

Use `bv` from PATH. It reads a project's JSONL export rather than its embedded
Dolt database.
As part of initializing each project, enable and seed that project's export:

```bash
BEADS_DIR="$sandbox_beads" bd config set export.auto true
BEADS_DIR="$sandbox_beads" bd config set export.path issues.jsonl
BEADS_DIR="$sandbox_beads" bd config set export.interval 1ms
BEADS_DIR="$sandbox_beads" bd export -o "$sandbox_beads/issues.jsonl"
```

The short export interval avoids the default throttle hiding recent issue
changes. Seed an export even for an empty tracker. After bulk deletions or
imports, refresh it explicitly so the viewer also reflects an empty project.
Keep JSONL passive: edit issues through `bd`, never through the export.

The user should then be able to run plain `bv` from the subproject root—the
directory containing its `.beads/`—without flags or a shell wrapper:

```bash
cd rl_math
bv
```

Do not persist global `BEADS_DIR` or `BEADS_DB` values in shell configuration;
both can override the current directory, and `BEADS_DB` takes precedence.
If such overrides are already present, clear them for ordinary local viewing.
Agents should use noninteractive commands such as `bv --robot-triage` when
checking viewer behavior.

For a feature worktree, create `<worktree>/<project>/.beads/redirect` containing
the absolute path of the project's `.beads/` in the primary checkout. First
verify that the local directory has no separate database or unrelated redirect.
Use an absolute target because tools can interpret relative redirects
differently. Keep the redirect machine-local and ignored by Git. Plain `bv`
from that worktree's project directory then follows the same export; do not
copy a database or maintain a second JSONL snapshot there. Continue using the
explicit workspace binding for `bd` operations.

## Default bead-to-PR workflow

Unless the user explicitly requests a different scope, complete one bead per
review cycle:

1. Select and verify the project's workspace as above. With that binding, run
   `bd prime`, inspect `git status`, check existing `in_progress` work, and run
   `bd ready`. Resume relevant work in its existing branch and PR rather
   than duplicating it. An explicit user request can be recorded as a new bead;
   do not invent additional roadmap scope when no ready work remains.
2. If no bead was specified, select the next logical ready task using
   dependencies, the experiment plan, priority, and a small coherent learning
   step. Read it with `bd show <id>` and claim it with
   `bd update <id> --claim`. Clarify only choices that materially change scope.
3. Fetch the remote default branch and create a dedicated worktree and
   bead-specific branch from its refreshed tip, for example
   `beads/<id>-<short-description>`. Keep one bead per worktree/branch and
   implement it there, preserving all unrelated work in the primary checkout.
   Point Beads at the selected project's primary-checkout database. Use a
   different or stacked base only when the user authorizes it, and explain
   the dependency in the PR.
4. Implement the acceptance criteria in clear, educational steps. For meaningful
   behavioral changes, add focused tests and use red/green/refactor where it
   helps establish correctness. For documentation or other low-impact work,
   use the closest useful deterministic validation; avoid tests that merely
   repeat the implementation.
5. Run the checks relevant to the changed experiment. There is no repository-wide
   application, test suite, or build command. Run short smoke experiments before
   longer training jobs and record seeds, settings, environment versions, and
   observed runtimes. Preserve user-created data and run outputs.
6. Commit only the bead's scoped changes, push its feature branch, and create
   or update a PR against the remote default branch. Reference the bead ID and
   describe the problem, resulting behavior, validation, and relevant figures
   or experiment results. This workflow authorizes these scoped commits,
   feature-branch pushes, and PR operations without another permission request.
   It does not authorize default-branch pushes, PR merges, auto-merge, or
   `bd dolt push`. Respect any narrower instruction from the current user.
7. Run applicable integration checks against the branch or PR. Fix in-scope
   failures and update the same PR before calling it ready for review. If no
   integration suite exists or a check could not run, say so in the handoff.
8. Append implementation notes with `bd update <id> --append-notes ...`,
   preserving existing notes. Record the owning workspace and prefix, branch or
   commit, PR URL, concise change summary, validation and integration results,
   and any limitations or follow-up.
   Leave the bead `in_progress` and mark it `awaiting-review` with
   `bd update <id> --add-label awaiting-review`.
9. Hand off the bead ID, change summary, checks, PR URL, and limitations. Stop
   for the user to review and merge manually. Do not merge the PR, close the
   bead, or start the next implementation bead as part of this review cycle.

## Merge verification and explicit closure approval

Close a bead only when both conditions hold:

* The hosting CLI or API confirms that its PR is merged.
* The user explicitly authorizes closing that bead.

A message that a PR was merged is not closure approval. If only merge is
confirmed, append that fact to the bead, remove `awaiting-review`, add
`awaiting-close-approval`, and keep it `in_progress`. An instruction such as
"merged, close this bead" provides closure approval; do not ask for it again.
Approval received before merge does not bypass merge verification.

After both conditions are satisfied:

1. Verify the PR state through the hosting CLI or API and record its URL and
   merge commit. If it is not merged, report the mismatch and leave the bead
   and feature worktree open.
2. Close with `bd close <id> --reason ...`, citing the merged PR, merge commit,
   and explicit user approval. Preserve implementation notes and remove review
   or closure-waiting labels. Refresh the passive issue export.
3. Fetch and prune remote refs, inspect the feature worktree and branches, and
   preserve unrelated changes and commits. This approval also authorizes cleanup
   of that merged bead's clean worktree and feature branches. Use normal safe
   removal first. Do not force removal or deletion without proving the worktree
   is clean and its work is represented in the merged PR/default branch; account
   for squash or rebase merges rather than assuming ancestry alone proves it.
4. Reconcile the local default branch only when it is safe without losing or
   rewriting unrelated work. Record closeout in Beads; publish any needed
   tracked metadata through a scoped branch and PR. Closing a bead grants no
   default-branch push, automatic merge, or Dolt remote-sync authority.
5. Report closure and cleanup results, any preserved worktrees or branches,
   relevant post-merge validation, and any local changes still needing review.

## Local tools and experiment conventions

Use `bd` from PATH. Project trackers use embedded Dolt and do not need a
separate database server for issue operations.

Keep committed instructions, PR descriptions, and bead notes portable: use
commands from PATH and repository-relative paths. Derive workspace paths at
runtime; keep machine-specific resolved paths only in ignored local state.

Use `python3` or the experiment's virtual-environment interpreter. Keep reusable
training code in Python modules and use notebooks to explain and visualize it.
Keep generated checkpoints, training outputs, and recordings out of Git;
selected figures for the post may be versioned with the experiment's sources.
Use non-interactive commands without overwriting unrelated user files.

## Beads persistence and handoff

Use `--json` for scripted queries, `bd update` flags instead of interactive
`bd edit`, and Beads dependencies for blockers. Follow-up tasks belong in Beads;
design documents describe experiments without duplicating live task status.
Use `bd remember` for project facts and Beads for all durable task tracking;
do not create markdown TODO lists or ad hoc project memory files.

The selected project's local Dolt database is authoritative. JSONL is a
passive viewer export, not a complete backup or the normal synchronization
mechanism. Do not edit it by hand or import it during ordinary Git operations.
Keep source-branch pushes separate from Beads remote synchronization.

Before handoff, refresh the project snapshot using the same explicit
workspace binding, check the intended worktree changes, and leave the bead
awaiting review:

```bash
BEADS_DIR="$sandbox_beads" bd export -o "$sandbox_beads/issues.jsonl"
git diff --check
git status --short --branch
```

No repository-local Beads skill or additional Beads README is required. A short
`PRIME.md` keeps hook-provided startup context pointed at this policy without
duplicating it. After choosing a subproject, refresh its explicitly bound
`bd prime` context even if a startup hook already ran from the repository root.
