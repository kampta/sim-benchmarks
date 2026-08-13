from __future__ import annotations

import numpy as np

from sim_benchmarks.model_servers.libero_null import LIBERONullModelServer


def test_libero_null_probe_returns_open_gripper_chunks() -> None:
    server = LIBERONullModelServer(chunk_size=10)
    action = server.predict({}, None)  # type: ignore[arg-type]
    actions = action["actions"]
    assert actions.shape == (10, 7)
    np.testing.assert_array_equal(actions[:, :6], 0.0)
    np.testing.assert_array_equal(actions[:, -1], -1.0)
    assert server.get_observation_params() == {"send_wrist_image": True, "send_state": True}
