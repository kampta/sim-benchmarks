"""Strict extensions for vla-eval."""

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
