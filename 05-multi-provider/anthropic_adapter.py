"""Translate OpenAI-style messages to Anthropic Messages API and back."""

from __future__ import annotations


def split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    rest = [m for m in messages if m.get("role") != "system"]
    system = "\n\n".join(str(p) for p in system_parts) if system_parts else None
    return system, rest


def content_blocks(content) -> list[dict]:
    if isinstance(content, list):
        blocks = []
        for part in content:
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text") or ""})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url") or ""
                if url.startswith("data:"):
                    header, b64 = url.split(",", 1)
                    media = "image/png"
                    if "image/jpeg" in header:
                        media = "image/jpeg"
                    elif "image/webp" in header:
                        media = "image/webp"
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media, "data": b64},
                        }
                    )
                else:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        return blocks or [{"type": "text", "text": ""}]
    return [{"type": "text", "text": content or ""}]


def to_anthropic_body(messages: list[dict], model: str, max_tokens: int, temperature: float) -> dict:
    system, rest = split_system(messages)
    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": m["role"], "content": content_blocks(m.get("content"))}
            for m in rest
            if m.get("role") in {"user", "assistant"}
        ],
    }
    if system:
        body["system"] = system
    return body


def from_anthropic_response(data: dict) -> dict:
    """Normalize Anthropic JSON to the OpenAI chat.completion shape our CLIs already print."""
    blocks = data.get("content") or []
    text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
    usage = data.get("usage") or {}
    stop = data.get("stop_reason")
    finish = {"end_turn": "stop", "max_tokens": "length"}.get(stop, stop)
    return {
        "id": data.get("id"),
        "object": "chat.completion",
        "model": data.get("model"),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        },
        "raw_anthropic": data,
    }
