"""Verify model artifacts against an immutable repository manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    calculate_hash: bool = True,
) -> dict[str, Any]:
    """Verify one immutable file and return a machine-readable check."""

    normalized_digest = expected_sha256.removeprefix("sha256:")
    check: dict[str, Any] = {
        "path": str(path),
        "expected_size": expected_size,
        "expected_sha256": normalized_digest,
        "exists": path.is_file(),
    }
    if not check["exists"]:
        raise RuntimeError(f"artifact verification failed: missing {path}")
    check["size"] = path.stat().st_size
    check["size_ok"] = check["size"] == expected_size
    if not check["size_ok"]:
        raise RuntimeError(f"artifact verification failed: size mismatch for {path}: {check['size']} != {expected_size}")
    if calculate_hash:
        check["sha256"] = _sha256(path)
        check["sha256_ok"] = check["sha256"] == normalized_digest
        if not check["sha256_ok"]:
            raise RuntimeError(f"artifact verification failed: SHA-256 mismatch for {path}")
    return check


def verify_checkpoint_files(
    root: Path,
    checkpoint: dict[str, Any],
    *,
    calculate_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Return per-file checks and fail closed on a missing or invalid artifact."""

    expected_files = checkpoint.get("weight_files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("checkpoint manifest does not declare weight_files")

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for expected in expected_files:
        path = root / expected["path"]
        expected_size = int(expected["size"])
        expected_sha256 = str(expected["sha256"])
        check: dict[str, Any] = {
            "path": str(path),
            "expected_size": expected_size,
            "expected_sha256": expected_sha256,
            "exists": path.is_file(),
        }
        if not path.is_file():
            errors.append(f"missing {path}")
            checks.append(check)
            continue

        check["size"] = path.stat().st_size
        check["size_ok"] = check["size"] == expected_size
        if not check["size_ok"]:
            errors.append(f"size mismatch for {path}: {check['size']} != {expected_size}")
        if calculate_hashes and check["size_ok"]:
            check["sha256"] = _sha256(path)
            check["sha256_ok"] = check["sha256"] == expected_sha256
            if not check["sha256_ok"]:
                errors.append(f"SHA-256 mismatch for {path}")
        checks.append(check)

    if errors:
        raise RuntimeError("checkpoint verification failed:\n" + "\n".join(errors))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="downloaded checkpoint directory")
    parser.add_argument("--manifest", type=Path, required=True, help="model manifest JSON")
    parser.add_argument("--benchmark", required=True, help="checkpoint benchmark key")
    parser.add_argument("--size-only", action="store_true", help="skip SHA-256 computation")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matches = [entry for entry in manifest.get("checkpoints", []) if entry.get("benchmark") == args.benchmark]
    if len(matches) != 1:
        raise ValueError(f"expected one checkpoint for benchmark {args.benchmark!r}, found {len(matches)}")
    checks = verify_checkpoint_files(args.root, matches[0], calculate_hashes=not args.size_only)
    print(json.dumps({"checkpoint": matches[0]["id"], "revision": matches[0]["revision"], "files": checks}, indent=2))


if __name__ == "__main__":
    main()
