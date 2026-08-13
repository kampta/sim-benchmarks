"""Verify the pinned VLABench OCI asset layer before extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim_benchmarks.provenance.artifacts import verify_file


def verify_vlabench_asset_layer(
    layer: Path,
    manifest: dict[str, Any],
    *,
    calculate_hash: bool = True,
) -> dict[str, Any]:
    """Verify the architecture-neutral layer declared by the benchmark manifest."""

    source = manifest.get("asset_source")
    if not isinstance(source, dict):
        raise TypeError("VLABench manifest does not declare an asset_source mapping")
    digest = source.get("asset_layer_digest")
    size = source.get("asset_layer_size")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("VLABench manifest does not declare a SHA-256 asset_layer_digest")
    if not isinstance(size, int) or size < 1:
        raise ValueError("VLABench manifest does not declare a positive asset_layer_size")
    return verify_file(
        layer,
        expected_size=size,
        expected_sha256=digest,
        calculate_hash=calculate_hash,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", type=Path, help="downloaded OCI layer tar.gz")
    parser.add_argument("--manifest", type=Path, required=True, help="VLABench benchmark manifest JSON")
    parser.add_argument("--size-only", action="store_true", help="skip SHA-256 computation")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    check = verify_vlabench_asset_layer(args.layer, manifest, calculate_hash=not args.size_only)
    print(json.dumps({"benchmark": manifest.get("id"), "asset_layer": check}, indent=2))


if __name__ == "__main__":
    main()
