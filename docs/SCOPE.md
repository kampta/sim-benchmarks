# Repository scope

`sim-benchmarks` evaluates robot policies in simulation. Every implementation
must contribute directly to selecting a benchmark episode, connecting a policy,
executing a rollout, measuring behavior, or reproducing and reporting a result.

## In scope

- simulation benchmark adapters and pinned benchmark runtimes;
- policy inference servers used as evaluation baselines;
- observation, action, timing, and embodiment-interface adapters;
- deterministic task, seed, episode, and perturbation manifests;
- benchmark-native metrics plus stratified statistical reports;
- synchronous and latency-aware rollout protocols;
- checkpoint, policy-training-data, normalization, asset, and container
  provenance needed to interpret a score or detect leakage;
- evaluation job orchestration and result comparison.

## Out of scope

- policy pretraining, post-training, fine-tuning, or dataset-mixture tooling;
- general dataset catalogs unrelated to an implemented evaluation suite;
- tokenizer, representation-learning, video-generation, or world-model
  experiments;
- evaluation whose output is generated-video quality rather than robot-policy
  behavior;
- physical-robot drivers, safety systems, teleoperation, or data collection;
- generic model serving not exercised by a benchmark runner.

Related experiments belong in separate repositories. A policy implementation
may live here only as the minimum inference adapter needed to evaluate a pinned
checkpoint. Benchmark-native demonstrations may be cataloged only for split
semantics, reference-policy reproduction, and train/evaluation collision checks.

## Acceptance test for new work

A proposed change belongs here only if all answers are yes:

1. Does it evaluate a robot policy on a named simulation benchmark?
2. Does it preserve or make explicit the benchmark's task, split, seed,
   embodiment, horizon, and metric protocol?
3. Does it produce or validate episode-level evidence that can support a
   benchmark result?

If not, it should be developed elsewhere and referenced here only through
immutable policy or provenance metadata when an evaluation requires it.
