from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sim_benchmarks.provenance.vlabench_assets import verify_vlabench_asset_layer


def _manifest(payload: bytes) -> dict[str, object]:
    return {
        "id": "vlabench",
        "asset_source": {
            "asset_layer_size": len(payload),
            "asset_layer_digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        },
    }


def test_verify_vlabench_asset_layer(tmp_path: Path) -> None:
    payload = b"immutable-vlabench-assets"
    layer = tmp_path / "layer.tar.gz"
    layer.write_bytes(payload)

    check = verify_vlabench_asset_layer(layer, _manifest(payload))

    assert check["size_ok"] is True
    assert check["sha256_ok"] is True


def test_verify_vlabench_asset_layer_rejects_wrong_digest(tmp_path: Path) -> None:
    payload = b"immutable-vlabench-assets"
    layer = tmp_path / "layer.tar.gz"
    layer.write_bytes(payload)
    manifest = _manifest(payload)
    manifest["asset_source"]["asset_layer_digest"] = f"sha256:{'0' * 64}"  # type: ignore[index]

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_vlabench_asset_layer(layer, manifest)
