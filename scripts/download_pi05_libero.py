"""Download and verify the pinned LeRobot pi0.5 LIBERO checkpoint."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_token, snapshot_download

REPO_ID = "lerobot/pi05_libero_finetuned_v044"
REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
DESTINATION = Path("/data/shared1/models/pi05_libero_finetuned_v044")
WEIGHT_SIZE = 7_473_096_344
WEIGHT_SHA256 = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
CHUNK_SIZE = 64 * 1024 * 1024
MAX_WORKERS = 8
RANGE_CACHE = Path("/data/shared1/cache/huggingface/downloads/pi05_libero_finetuned_v044")
WEIGHT_URL = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/model.safetensors?download=true"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_range(index: int, start: int, end: int, token: str | None) -> Path:
    RANGE_CACHE.mkdir(parents=True, exist_ok=True)
    destination = RANGE_CACHE / f"chunk-{index:04d}-{start}-{end}"
    expected = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected:
        return destination

    attempt = 0
    while True:
        attempt += 1
        temporary = destination.with_suffix(".tmp")
        completed = temporary.stat().st_size if temporary.is_file() else 0
        if completed > expected:
            temporary.unlink()
            completed = 0
        if completed == expected:
            temporary.replace(destination)
            return destination
        request_start = start + completed
        headers = {"Range": f"bytes={request_start}-{end}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with requests.get(WEIGHT_URL, headers=headers, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                if request_start > 0 and response.status_code != requests.codes.partial_content:
                    raise OSError(f"range {index} returned HTTP {response.status_code}, expected 206")
                written = completed
                with temporary.open("ab") as handle:
                    for block in response.iter_content(1024 * 1024):
                        remaining = expected - written
                        if remaining <= 0:
                            break
                        block = block[:remaining]
                        handle.write(block)
                        written += len(block)
                if written != expected:
                    raise OSError(f"range {index} expected {expected} bytes, received {written}")
            temporary.replace(destination)
            print(f"downloaded range {index}: bytes {start}-{end}", flush=True)
            return destination
        except (OSError, requests.RequestException) as exc:
            print(f"range {index} attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=DESTINATION,
        allow_patterns=["*.json", "*.md", ".gitattributes", "*processor*.safetensors"],
        max_workers=4,
    )

    weights = DESTINATION / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size != WEIGHT_SIZE:
        ranges = [
            (index, start, min(start + CHUNK_SIZE, WEIGHT_SIZE) - 1)
            for index, start in enumerate(range(0, WEIGHT_SIZE, CHUNK_SIZE))
        ]
        token = get_token()
        paths: dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_download_range, index, start, end, token): index
                for index, start, end in ranges
            }
            for future in as_completed(futures):
                paths[futures[future]] = future.result()

        temporary = weights.with_suffix(".safetensors.assembling")
        with temporary.open("wb") as output:
            for index, _, _ in ranges:
                with paths[index].open("rb") as source:
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != WEIGHT_SIZE:
            raise RuntimeError(f"assembled checkpoint has invalid size {temporary.stat().st_size}")
        temporary.replace(weights)

    actual = _sha256(weights)
    if actual != WEIGHT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: expected {WEIGHT_SHA256}, got {actual}")
    print(f"verified {weights} sha256={actual}", flush=True)


if __name__ == "__main__":
    main()
