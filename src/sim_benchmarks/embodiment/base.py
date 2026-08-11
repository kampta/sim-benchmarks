"""Interfaces for auditable observation/action conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from sim_benchmarks.protocol.negotiation import NegotiationResult, require_compatible
from sim_benchmarks.protocol.specs import EndpointSpec


class EmbodimentAdapter(ABC):
    """A named, versioned conversion between benchmark and policy contracts.

    The adapter exposes the benchmark as seen by the policy. This prevents
    camera remapping, normalization, frame transforms, and gripper inversions
    from being hidden inside model-specific inference code.
    """

    adapter_id: str
    version: str

    @abstractmethod
    def policy_facing_spec(self, benchmark_spec: EndpointSpec) -> EndpointSpec:
        """Return the exact interface exposed after transformation."""

    @abstractmethod
    def observation_to_policy(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one canonical benchmark observation for policy inference."""

    @abstractmethod
    def action_to_benchmark(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one policy action to the benchmark controller contract."""

    def negotiate(self, benchmark_spec: EndpointSpec, policy_spec: EndpointSpec) -> NegotiationResult:
        return require_compatible(self.policy_facing_spec(benchmark_spec), policy_spec)


class IdentityEmbodimentAdapter(EmbodimentAdapter):
    adapter_id = "identity"
    version = "1.0"

    def policy_facing_spec(self, benchmark_spec: EndpointSpec) -> EndpointSpec:
        return benchmark_spec

    def observation_to_policy(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return dict(observation)

    def action_to_benchmark(self, action: Mapping[str, Any]) -> dict[str, Any]:
        return dict(action)
