# Reproducibility requirements

## Support states

- `planned`: source selected and pinned.
- `scaffolded`: adapter and container build exist.
- `smoke-tested`: deterministic reset plus at least one successful step.
- `reproduced`: a released checkpoint matches an upstream reference within the
  declared tolerance using the same episode protocol.
- `supported`: reproduced, regression-tested, documented, and included in a
  maintained suite.

Only `reproduced` and `supported` are public result-bearing states.

## Required run provenance

Every result must record:

- harness, extension, benchmark, asset, and container revisions;
- checkpoint identifier and immutable digest;
- training dataset mixture and normalization-statistics digest;
- embodiment-adapter identifier and version;
- benchmark suite, split, task list, seed list, and episode-manifest digest;
- observation/action interface manifests after negotiation;
- synchronous or live execution mode and control frequency;
- requested versus executed action horizon;
- simulator, inference, and wall-clock timing;
- episode failures and retries without silently replacing seeds.

## Statistical report

Report task and stage success, bootstrap confidence intervals, paired episode
comparisons, latency percentiles, safety violations, and failures. ID and OOD
conditions must be separated. A global average may supplement but never replace
the stratified report.

## Data boundaries

Use the three-way policy in the dataset catalog:

- `train_id`: training is allowed;
- `eval_id`: same distribution, unseen episode instances;
- `eval_ood`: withheld generalization factors.

An episode identifier, initial-state seed, scene, object instance, or generated
variation in an evaluation manifest must not appear in training. Dataset and
evaluation manifests are compared by content hash before a run is accepted.
Training data is recorded only as policy provenance for this check; policy
training and general-purpose dataset tooling are outside this repository.
