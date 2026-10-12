from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests


LISTING_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{release}/output/evidence_clinical_precedence/"
DOWNLOAD_BASE = "https://open-targets-public-data-releases.s3.amazonaws.com/platform/{release}/output/evidence_clinical_precedence/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_release(release: str, root: Path) -> list[Path]:
    listing_url = LISTING_BASE.format(release=release)
    download_url = DOWNLOAD_BASE.format(release=release)
    response = requests.get(listing_url, timeout=60)
    response.raise_for_status()
    names = re.findall(r'href="([^"?]+\.parquet)"', response.text)
    if not names:
        raise RuntimeError(f"No parquet files found at {listing_url}")
    destination = root / release
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, name in enumerate(names, 1):
        path = destination / name
        paths.append(path)
        expected = int(requests.head(download_url + name, timeout=60).headers["Content-Length"])
        if path.exists() and path.stat().st_size == expected:
            print(f"[{release}] {index}/{len(names)} cached {name}", flush=True)
            continue
        print(f"[{release}] {index}/{len(names)} downloading {name}", flush=True)
        chunk_size = 4 * 1024 * 1024
        with path.open("wb") as target:
            for start in range(0, expected, chunk_size):
                end = min(expected - 1, start + chunk_size - 1)
                source = requests.get(
                    download_url + name,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 120),
                )
                source.raise_for_status()
                payload = source.content
                if len(payload) != end - start + 1:
                    raise RuntimeError(
                        f"Unexpected range length for {name}: {len(payload)} bytes for {start}-{end}"
                    )
                target.write(payload)
                print(f"[{release}] {name}: {end + 1:,}/{expected:,} bytes", flush=True)
        if path.stat().st_size != expected:
            raise RuntimeError(
                f"Incomplete download for {name}: {path.stat().st_size} bytes, expected {expected}"
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--releases", nargs="+", default=["26.03", "26.06"])
    args = parser.parse_args()
    manifest = []
    for release in args.releases:
        paths = download_release(release, args.output)
        first = pd.read_parquet(paths[0])
        print(f"[{release}] columns={list(first.columns)}", flush=True)
        print(first.head(2).to_string(), flush=True)
        for path in paths:
            manifest.append(
                {
                    "release": release,
                    "url": DOWNLOAD_BASE.format(release=release) + path.name,
                    "file": str(path.relative_to(args.output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (args.output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
