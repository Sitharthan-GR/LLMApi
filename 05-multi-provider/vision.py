#!/usr/bin/env python3
"""Caption a local image via OpenAI-style image_url content parts.

  python 05-multi-provider/vision.py
  python 05-multi-provider/vision.py --image 05-multi-provider/sample.png
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from client import ChatClient
from profiles import get_profile, load_env


def write_red_png(path: Path, width: int = 48, height: int = 48) -> None:
    """Tiny PNG with no extra deps. The caption task is 'what color?'."""
    rgb = (220, 32, 32)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + (bytes(rgb) * width) for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )


def data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--image", default=str(HERE / "sample.png"))
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    load_env(ROOT)
    profile = get_profile(args.provider)
    model = args.model or profile.vision_model
    if not model:
        sys.exit(f"{profile.name} has no vision_model in profiles.py")

    image_path = Path(args.image)
    if not image_path.exists() and image_path.resolve() == (HERE / "sample.png").resolve():
        write_red_png(image_path)
        print(f"wrote {image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Return JSON only: {\"color\": \"...\", \"shape\": \"...\"} describing this image.",
                },
                {"type": "image_url", "image_url": {"url": data_url(image_path)}},
            ],
        }
    ]
    print(f"provider={profile.name} model={model}")
    print(f"image={image_path}  (content[1].type=image_url)")
    client = ChatClient(profile, timeout=120.0)
    data = client.complete(messages, model=model, temperature=0.1, max_tokens=args.max_tokens)
    (HERE / "last-vision-response.json").write_text(json.dumps(data, indent=2) + "\n")
    print("assistant>")
    print(data["choices"][0]["message"].get("content") or "")


if __name__ == "__main__":
    main()
