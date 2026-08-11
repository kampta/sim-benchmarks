# Upstream policy

## Pinned base

- Repository: `https://github.com/allenai/vla-evaluation-harness.git`
- Tag: `v0.4.0`
- Commit: `2680ab2fafe981c2dba63c6c1a4e7bb4415dbb56`
- Pin date: `2026-08-11`

The exact commit is declared in `pyproject.toml`; floating branches and
`latest` Docker tags are not valid inputs for published evaluations.

## Fork model

This repository is a thin extension, not a source copy of the harness.
Generic runner, protocol, metrics, and correctness fixes should be proposed
upstream. Benchmark-specific adapters, suite composition, private cluster
orchestration, and local reproduction evidence remain here.

The local Git remote named `upstream` points to AllenAI. Creating a hosted
GitHub fork remains an administrative step because this environment has no
authenticated GitHub account.

## Updating

An upstream update requires all of the following:

1. Change the full commit SHA in `pyproject.toml` and this file.
2. Re-lock dependencies.
3. Run unit and protocol-conformance tests.
4. Re-run every reproduction target marked `verified`.
5. Record any score or rollout change before accepting the update.

