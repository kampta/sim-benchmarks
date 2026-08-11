"""Fail-closed negotiation between benchmark and policy endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from sim_benchmarks.protocol.specs import ActionField, EndpointSpec, ObservationField, shape_matches


@dataclass(frozen=True)
class NegotiationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class NegotiationResult:
    compatible: bool
    mode: str | None
    issues: tuple[NegotiationIssue, ...]


class InterfaceNegotiationError(RuntimeError):
    def __init__(self, result: NegotiationResult) -> None:
        self.result = result
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        super().__init__(f"policy/benchmark interface negotiation failed: {details}")


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def _compare_observation(produced: ObservationField, required: ObservationField) -> list[tuple[str, str]]:
    mismatches: list[tuple[str, str]] = []
    checks = (
        ("kind", produced.kind, required.kind),
        ("dtype", produced.dtype, required.dtype),
        ("encoding", produced.encoding, required.encoding),
        ("unit", produced.unit, required.unit),
        ("frame", produced.frame, required.frame),
    )
    for label, got, want in checks:
        if want and got != want:
            mismatches.append((label, f"produces {got!r}, policy requires {want!r}"))
    if not shape_matches(produced.shape, required.shape):
        mismatches.append(("shape", f"produces {produced.shape}, policy requires {required.shape}"))
    return mismatches


def _compare_action(produced: ActionField, consumed: ActionField) -> list[tuple[str, str]]:
    mismatches: list[tuple[str, str]] = []
    checks = (
        ("dtype", produced.dtype, consumed.dtype),
        ("representation", produced.representation, consumed.representation),
        ("unit", produced.unit, consumed.unit),
        ("frame", produced.frame, consumed.frame),
    )
    for label, got, want in checks:
        if want and got != want:
            mismatches.append((label, f"policy produces {got!r}, benchmark consumes {want!r}"))
    if not shape_matches(produced.shape, consumed.shape):
        mismatches.append(("shape", f"policy produces {produced.shape}, benchmark consumes {consumed.shape}"))
    return mismatches


def negotiate(benchmark: EndpointSpec, policy: EndpointSpec, requested_mode: str | None = None) -> NegotiationResult:
    """Negotiate a rollout contract after any embodiment adapter is applied.

    `benchmark` describes observations produced and actions consumed. `policy`
    describes observations consumed and actions produced. Any mismatch is an
    error; conversions must be made explicit before this function is called.
    """

    issues: list[NegotiationIssue] = []
    if _major(benchmark.protocol_version) != _major(policy.protocol_version):
        issues.append(
            NegotiationIssue(
                "protocol_version",
                "protocol_version",
                f"benchmark {benchmark.protocol_version} is incompatible with policy {policy.protocol_version}",
            )
        )
    if policy.embodiment not in {benchmark.embodiment, "*"}:
        issues.append(
            NegotiationIssue(
                "embodiment",
                "embodiment",
                f"benchmark is {benchmark.embodiment!r}, policy is {policy.embodiment!r}",
            )
        )

    common_modes = benchmark.modes & policy.modes
    mode: str | None = None
    if requested_mode is not None:
        if requested_mode not in common_modes:
            issues.append(NegotiationIssue("execution_mode", "modes", f"{requested_mode!r} is not mutually supported"))
        else:
            mode = requested_mode
    elif common_modes:
        mode = "sync" if "sync" in common_modes else min(common_modes)
    else:
        issues.append(NegotiationIssue("execution_mode", "modes", "no mutually supported execution mode"))

    if abs(benchmark.timing.control_hz - policy.timing.control_hz) > 1e-6:
        issues.append(
            NegotiationIssue(
                "control_rate",
                "timing.control_hz",
                f"benchmark is {benchmark.timing.control_hz} Hz, policy is {policy.timing.control_hz} Hz",
            )
        )
    if abs(benchmark.timing.observation_hz - policy.timing.observation_hz) > 1e-6:
        issues.append(
            NegotiationIssue(
                "observation_rate",
                "timing.observation_hz",
                f"benchmark is {benchmark.timing.observation_hz} Hz, policy is {policy.timing.observation_hz} Hz",
            )
        )
    if benchmark.timing.action_horizon != policy.timing.action_horizon:
        issues.append(
            NegotiationIssue(
                "action_horizon",
                "timing.action_horizon",
                f"benchmark wire contract is {benchmark.timing.action_horizon}, "
                f"policy wire contract is {policy.timing.action_horizon}",
            )
        )
    if policy.timing.observation_history > benchmark.timing.observation_history:
        issues.append(
            NegotiationIssue(
                "observation_history",
                "timing.observation_history",
                f"benchmark supplies {benchmark.timing.observation_history}, policy needs {policy.timing.observation_history}",
            )
        )

    benchmark_observations = benchmark.observation_map
    for name, required in policy.observation_map.items():
        produced = benchmark_observations.get(name)
        if produced is None:
            if required.required:
                issues.append(NegotiationIssue("missing_observation", f"observations.{name}", "required by policy"))
            continue
        for field, message in _compare_observation(produced, required):
            issues.append(NegotiationIssue("observation_mismatch", f"observations.{name}.{field}", message))

    policy_actions = policy.action_map
    for name, consumed in benchmark.action_map.items():
        produced = policy_actions.get(name)
        if produced is None:
            if consumed.required:
                issues.append(NegotiationIssue("missing_action", f"actions.{name}", "required by benchmark"))
            continue
        for field, message in _compare_action(produced, consumed):
            issues.append(NegotiationIssue("action_mismatch", f"actions.{name}.{field}", message))

    return NegotiationResult(compatible=not issues, mode=mode if not issues else None, issues=tuple(issues))


def require_compatible(
    benchmark: EndpointSpec, policy: EndpointSpec, requested_mode: str | None = None
) -> NegotiationResult:
    result = negotiate(benchmark, policy, requested_mode=requested_mode)
    if not result.compatible:
        raise InterfaceNegotiationError(result)
    return result
