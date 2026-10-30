"""Remember the user's chart settings. Values come back from the page URL or the owner's private profile, so they are
UNTRUSTED input: everything is validated against a schema and anything unexpected falls back to the default."""
from __future__ import annotations

import base64
import json
import zlib

MAX_ENCODED = 3000


# ---- validators
def choice(options):
    opts = list(options)
    return lambda v: v in opts


def multi(options):
    opts = set(options)
    return lambda v: isinstance(v, list) and len(v) <= 40 and all(isinstance(x, str) and x in opts for x in v)


def num(lo, hi):
    return lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def flag():
    return lambda v: isinstance(v, bool)


def clean(values: dict | None, schema: dict) -> dict:
    """schema = {key: (default, validator)}. Unknown keys are dropped, invalid values replaced by the default."""
    values = values if isinstance(values, dict) else {}
    out = {}
    for key, (default, ok) in schema.items():
        v = values.get(key, default)
        try:
            v = v if ok(v) else default
        except Exception:
            v = default
        if isinstance(v, (int, float)) and not isinstance(v, bool):          # widgets want the same number type as their default
            if isinstance(default, float):
                v = float(v)
            elif isinstance(default, int) and float(v).is_integer():
                v = int(v)
        out[key] = v
    return out


# ---- URL encoding (compact, URL-safe)
def encode(values: dict) -> str:
    raw = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


def decode(text: str | None) -> dict | None:
    if not text or len(text) > MAX_ENCODED:
        return None
    try:
        raw = zlib.decompressobj().decompress(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)), 20_000)   # size-capped
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
