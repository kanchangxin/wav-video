#!/usr/bin/env python3
"""Small local-only client for the Draw Things app HTTP API."""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7860"


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="PNG/JPEG input for img2img")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--sampler", default="Euler a")
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance,
        "strength": args.strength,
        "sampler": args.sampler,
        "shift": args.shift,
        "seed": args.seed,
        "hires_fix": False,
        "batch_count": 1,
        "batch_size": 1,
    }
    endpoint = "/sdapi/v1/txt2img"
    if args.input:
        payload["init_images"] = [base64.b64encode(args.input.read_bytes()).decode("ascii")]
        endpoint = "/sdapi/v1/img2img"

    start = time.monotonic()
    try:
        result = request(endpoint, payload)
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8", errors="replace"))
        raise
    elapsed = time.monotonic() - start
    images = result.get("images", [])
    if not images:
        raise RuntimeError(f"No images in response: {str(result)[:500]}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(images[0].split(",")[-1]))
    record = {"endpoint": endpoint, "parameters": {k: v for k, v in payload.items() if k != "init_images"},
              "input": str(args.input) if args.input else None, "output": str(output),
              "elapsed_seconds": round(elapsed, 2), "bytes": output.stat().st_size}
    output.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
