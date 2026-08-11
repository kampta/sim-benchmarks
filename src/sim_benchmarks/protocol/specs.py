"""Strict observation, action, and timing specifications.

`vla-eval` v0.4.0 intentionally uses permissive dictionaries and warns about
many mismatches. These immutable contracts form the stricter layer used by
this project before any rollout begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PROTOCOL_VERSION = "2.0"

ObservationKind = Literal["image", "state", "language", "tactile", "audio"]
ExecutionMode = Literal["sync", "live"]


def _validate_shape(name: str, shape: tuple[int, ...]) -> None:
    if any(dim == 0 or dim < -1 for dim in shape):
        raise ValueError(f"{name}: shape dimensions must be positive or -1 wildcards")


def _shape_matches(produced: tuple[int, ...], required: tuple[int, ...]) -> bool:
    return len(produced) == len(required) and all(want == -1 or got == want for got, want in zip(produced, required))


@dataclass(frozen=True)
class ObservationField:
    name: str
    kind: ObservationKind
    shape: tuple[int, ...] = ()
    dtype: str = ""
    encoding: str = ""
    unit: str = ""
    frame: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation field name cannot be empty")
        _validate_shape(self.name, self.shape)
        if self.kind != "language" and not self.dtype:
            raise ValueError(f"{self.name}: dtype is required for {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ObservationField:
        return cls(**{**raw, "shape": tuple(raw.get("shape", ()))})


@dataclass(frozen=True)
class ActionField:
    name: str
    shape: tuple[int, ...]
    dtype: str
    representation: str
    unit: str = ""
    frame: str = ""
    minimum: tuple[float, ...] | None = None
    maximum: tuple[float, ...] | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.representation or not self.dtype:
            raise ValueError("action name, dtype, and representation are required")
        _validate_shape(self.name, self.shape)
        size = 1
        for dim in self.shape:
            if dim == -1:
                size = -1
                break
            size *= dim
        for label, bound in (("minimum", self.minimum), ("maximum", self.maximum)):
            if bound is not None and size != -1 and len(bound) not in (1, size):
                raise ValueError(f"{self.name}: {label} must contain one or {size} values")
        if self.minimum is not None and self.maximum is not None:
            lower = self.minimum if len(self.minimum) > 1 else self.minimum * max(size, 1)
            upper = self.maximum if len(self.maximum) > 1 else self.maximum * max(size, 1)
            if any(lo >= hi for lo, hi in zip(lower, upper)):
                raise ValueError(f"{self.name}: every minimum must be less than maximum")

    def to_dict(self) -> dict[str, Any]:
        raw = {**self.__dict__, "shape": list(self.shape)}
        if self.minimum is not None:
            raw["minimum"] = list(self.minimum)
        if self.maximum is not None:
            raw["maximum"] = list(self.maximum)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ActionField:
        values = {**raw, "shape": tuple(raw["shape"])}
        if raw.get("minimum") is not None:
            values["minimum"] = tuple(raw["minimum"])
        if raw.get("maximum") is not None:
            values["maximum"] = tuple(raw["maximum"])
        return cls(**values)


@dataclass(frozen=True)
class TimingSpec:
    control_hz: float
    observation_hz: float
    action_horizon: int = 1
    observation_history: int = 1

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.observation_hz <= 0:
            raise ValueError("control and observation rates must be positive")
        if self.action_horizon < 1 or self.observation_history < 1:
            raise ValueError("action horizon and observation history must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class EndpointSpec:
    endpoint_id: str
    embodiment: str
    observations: tuple[ObservationField, ...]
    actions: tuple[ActionField, ...]
    timing: TimingSpec
    modes: frozenset[ExecutionMode] = field(default_factory=lambda: frozenset({"sync"}))
    protocol_version: str = PROTOCOL_VERSION
    recurrent_reset: bool = False

    def __post_init__(self) -> None:
        if not self.endpoint_id or not self.embodiment:
            raise ValueError("endpoint_id and embodiment are required")
        if not self.modes:
            raise ValueError("at least one execution mode is required")
        invalid_modes = self.modes - {"sync", "live"}
        if invalid_modes:
            raise ValueError(f"invalid execution modes: {sorted(invalid_modes)}")
        for label, values in (("observation", self.observations), ("action", self.actions)):
            names = [value.name for value in values]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label} field names")

    @property
    def observation_map(self) -> dict[str, ObservationField]:
        return {field.name: field for field in self.observations}

    @property
    def action_map(self) -> dict[str, ActionField]:
        return {field.name: field for field in self.actions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "embodiment": self.embodiment,
            "protocol_version": self.protocol_version,
            "observations": [value.to_dict() for value in self.observations],
            "actions": [value.to_dict() for value in self.actions],
            "timing": self.timing.to_dict(),
            "modes": sorted(self.modes),
            "recurrent_reset": self.recurrent_reset,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EndpointSpec:
        return cls(
            endpoint_id=raw["endpoint_id"],
            embodiment=raw["embodiment"],
            protocol_version=raw.get("protocol_version", PROTOCOL_VERSION),
            observations=tuple(ObservationField.from_dict(value) for value in raw["observations"]),
            actions=tuple(ActionField.from_dict(value) for value in raw["actions"]),
            timing=TimingSpec(**raw["timing"]),
            modes=frozenset(raw.get("modes", ["sync"])),
            recurrent_reset=raw.get("recurrent_reset", False),
        )


def shape_matches(produced: tuple[int, ...], required: tuple[int, ...]) -> bool:
    """Return whether a produced shape satisfies a required shape."""

    return _shape_matches(produced, required)
