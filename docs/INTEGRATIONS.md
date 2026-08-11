# First-wave integration contracts

Every source below is pinned in `src/sim_benchmarks/manifests/benchmarks`.
The implementation must preserve upstream task enumeration, initial states,
horizons, metrics, and train/test splits.

## Colosseum V2

Use a normal `vla_eval.benchmarks.base.StepBenchmark` adapter around its
ManiSkill environment. The adapter must expose the perturbation identifier and
factor as task metadata, aggregate by task and perturbation, and retain the
upstream clean-condition reference. Do not average all perturbations into one
number without reporting the per-factor profile.

Exit criteria:

- deterministic task/episode enumeration;
- clean and perturbed smoke tests;
- released ACT checkpoint reproduction;
- one LeRobot policy reproduction;
- CPU/GPU rendering declared only after measurement.

## VLA-Arena

Use a normal step adapter when the selected task family exposes an environment
loop. Preserve level, suite, safety category, distractor condition, and
generalization axis in every episode record. Export safety violations,
state-preservation failures, stage completion, and task success separately.

Exit criteria:

- at least one task from every requested capability category;
- successful RLDS/LeRobot observation and action mapping;
- reference reproduction for one OpenPI-family and one OpenVLA-family model;
- validation that a task cannot pass through a shortcut/no-language probe.

## DOMINO

DOMINO already separates its RoboTwin-derived simulation process and policy
process with WebSockets. Replace its policy transport with the pinned
`vla-eval` connection while leaving the upstream environment, episode manifest,
and metric implementation authoritative. Support two suites:

- synchronous reproduction;
- live evaluation with measured inference latency and stale-action ratio.

Preserve both upstream Success Rate and Manipulation Score, including collision
and out-of-bounds penalties. Never regenerate accepted episode seeds during a
model comparison; use the canonical screened manifest.

Exit criteria:

- clean-static, clean-dynamic, and randomized-dynamic task coverage;
- canonical episode manifest hash recorded in results;
- PUMA reference reproduction;
- deliberate latency injection test demonstrating live-mode degradation.

## EBench

EBench's topology differs from ordinary harness adapters: GenManip owns the
Isaac Sim service and task stream, while model code uses `EvalClient`. Implement
a custom `EpisodeRunner` that bridges GenManip observations/actions to the
`vla-eval` policy WebSocket. Do not pretend EBench is a local `StepBenchmark` or
replace its held-out online evaluation.

The runner must preserve:

- Specialist and Generalist tracks;
- the train/validation/test split selected by the EBench run;
- all five capability and four generalization dimensions;
- upstream worker/run identifiers and diagnostic artifacts;
- joint-position, gripper, and mobile-base action semantics.

Exit criteria:

- local validation run through the GenManip service;
- exact action-chunk behavior and early-stop handling;
- one published baseline reproduction, preferably pi0.5;
- held-out online submission path kept separate from local debugging.

