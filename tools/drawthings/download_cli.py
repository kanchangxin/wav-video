#!/usr/bin/env python3
"""Resume the official Draw Things CLI release with parallel HTTP ranges."""

import concurrent.futures
import os
import time
import urllib.request
from pathlib import Path

URL = "https://github.com/drawthingsai/draw-things-community/releases/download/v26.1023.0/draw-things-cli"
TOTAL = 290_110_032
CHUNK = 1_048_576
DEST = Path(__file__).with_name("draw-things-cli")
PARTS = Path(__file__).with_name(".cli-parts")


def fetch(part):
    start, end = part
    target = PARTS / f"{start:09d}.part"
    if target.exists() and target.stat().st_size == end - start + 1:
        return target
    for attempt in range(4):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=180) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"Unexpected HTTP range response: {response.status} {content_range}")
                data = response.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"Short range: {start}-{end}, got {len(data)} bytes")
            target.write_bytes(data)
            return target
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def main():
    PARTS.mkdir(exist_ok=True)
    prefix = DEST.stat().st_size if DEST.exists() else 0
    if prefix > TOTAL:
        raise RuntimeError(f"Existing file is too large: {prefix}")
    parts = [(start, min(start + CHUNK, TOTAL) - 1) for start in range(prefix, TOTAL, CHUNK)]
    print(f"Resuming from {prefix:,} of {TOTAL:,} bytes, {len(parts)} ranges", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(fetch, part): part for part in parts}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 20 == 0 or completed == len(parts):
                print(f"Downloaded {completed}/{len(parts)} ranges", flush=True)
    with DEST.open("ab") as output:
        for start, _ in parts:
            with (PARTS / f"{start:09d}.part").open("rb") as source:
                while data := source.read(CHUNK):
                    output.write(data)
    if DEST.stat().st_size != TOTAL:
        raise RuntimeError(f"Assembled size mismatch: {DEST.stat().st_size}")
    os.chmod(DEST, 0o755)
    print(f"Assembled {DEST} ({TOTAL:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
