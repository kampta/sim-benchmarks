from __future__ import annotations

import unittest

from sim_benchmarks.embodiment import IdentityEmbodimentAdapter
from sim_benchmarks.protocol.negotiation import InterfaceNegotiationError, negotiate
from sim_benchmarks.protocol.specs import ActionField, EndpointSpec, ObservationField, TimingSpec


def endpoint(*, endpoint_id: str, action_representation: str = "delta_xyz", image_shape=(224, 224, 3)) -> EndpointSpec:
    return EndpointSpec(
        endpoint_id=endpoint_id,
        embodiment="franka",
        observations=(
            ObservationField("main", "image", image_shape, "uint8", encoding="rgb_hwc"),
            ObservationField("instruction", "language"),
        ),
        actions=(
            ActionField(
                "eef_position",
                (3,),
                "float32",
                action_representation,
                unit="normalized",
                frame="robot_base",
            ),
        ),
        timing=TimingSpec(control_hz=20, observation_hz=20),
        modes=frozenset({"sync", "live"}),
    )


class ProtocolTests(unittest.TestCase):
    def test_identical_specs_negotiate(self) -> None:
        result = negotiate(endpoint(endpoint_id="benchmark"), endpoint(endpoint_id="policy"))
        self.assertTrue(result.compatible)
        self.assertEqual(result.mode, "sync")
        self.assertFalse(result.issues)

    def test_action_convention_mismatch_fails_closed(self) -> None:
        result = negotiate(
            endpoint(endpoint_id="benchmark", action_representation="delta_xyz"),
            endpoint(endpoint_id="policy", action_representation="absolute_xyz"),
        )
        self.assertFalse(result.compatible)
        self.assertTrue(any(issue.path == "actions.eef_position.representation" for issue in result.issues))

    def test_image_shape_mismatch_fails_closed(self) -> None:
        result = negotiate(
            endpoint(endpoint_id="benchmark", image_shape=(256, 256, 3)),
            endpoint(endpoint_id="policy", image_shape=(224, 224, 3)),
        )
        self.assertFalse(result.compatible)
        self.assertTrue(any(issue.path == "observations.main.shape" for issue in result.issues))

    def test_timing_mismatch_fails_closed(self) -> None:
        benchmark = endpoint(endpoint_id="benchmark")
        policy = EndpointSpec(
            endpoint_id="policy",
            embodiment=benchmark.embodiment,
            observations=benchmark.observations,
            actions=benchmark.actions,
            timing=TimingSpec(control_hz=20, observation_hz=10, action_horizon=4),
            modes=benchmark.modes,
        )
        result = negotiate(benchmark, policy)
        self.assertFalse(result.compatible)
        self.assertTrue(any(issue.code == "observation_rate" for issue in result.issues))
        self.assertTrue(any(issue.code == "action_horizon" for issue in result.issues))

    def test_identity_adapter_preserves_contract_and_payload(self) -> None:
        adapter = IdentityEmbodimentAdapter()
        benchmark = endpoint(endpoint_id="benchmark")
        policy = endpoint(endpoint_id="policy")
        self.assertTrue(adapter.negotiate(benchmark, policy).compatible)
        observation = {"main": "frame", "instruction": "pick"}
        action = {"eef_position": [0.0, 0.0, 0.0]}
        self.assertEqual(adapter.observation_to_policy(observation), observation)
        self.assertEqual(adapter.action_to_benchmark(action), action)

    def test_adapter_raises_for_incompatible_contract(self) -> None:
        adapter = IdentityEmbodimentAdapter()
        with self.assertRaises(InterfaceNegotiationError):
            adapter.negotiate(
                endpoint(endpoint_id="benchmark"),
                endpoint(endpoint_id="policy", action_representation="absolute_xyz"),
            )
