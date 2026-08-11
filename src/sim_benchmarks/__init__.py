"""Strict simulation-benchmark evaluation for robot policies."""

from sim_benchmarks.protocol.negotiation import (
    InterfaceNegotiationError,
    NegotiationResult,
    negotiate,
    require_compatible,
)
from sim_benchmarks.protocol.specs import ActionField, EndpointSpec, ObservationField, TimingSpec

__all__ = [
    "ActionField",
    "EndpointSpec",
    "InterfaceNegotiationError",
    "NegotiationResult",
    "ObservationField",
    "TimingSpec",
    "negotiate",
    "require_compatible",
]
