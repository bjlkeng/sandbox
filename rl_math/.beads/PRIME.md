# Beads workflow policy

Read the repository-root `AGENTS.md` and any instructions for the selected
subproject. They define the workflow; do not substitute generic completion rules.

Select and verify the subproject's own primary-checkout database with an explicit
`BEADS_DIR` before issue operations. The root tracker is for repository maintenance.
Use one bead per worktree, branch, and PR. Leave it `in_progress` for manual user
review and merge. Close it only after verifying merge and receiving the user's
explicit approval to close. Never merge PRs or close beads automatically.
