import re
import unicodedata

_SCRIPT_RANGES = {
    "kn": (0x0C80, 0x0CFF), "hi": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF), "latin": (0x0041, 0x007A),
}

def script_share(text: str, low: int, high: int) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(low <= ord(c) <= high for c in letters) / len(letters)

def extract_features(text: str) -> dict:
    scripts = {name: script_share(text, lo, hi) for name, (lo, hi) in _SCRIPT_RANGES.items()}
    digits = sum(c.isdigit() for c in text)
    words = max(1, len(text.split()))
    return {
        "length": len(text),
        "word_count": words,
        "script_mix": sum(v > 0 for v in scripts.values()),
        "digit_share": digits / max(1, len(text)),
        "named_entity_density": len(re.findall(r"\b[A-Z][A-Za-z0-9._-]+\b", text)) / words,
        "code_mixing": scripts["latin"] * sum(v for k, v in scripts.items() if k != "latin") > 0,
        "latin_share": scripts["latin"],
        "kannada_share": scripts["kn"],
        "hindi_share": scripts["hi"],
        "tamil_share": scripts["ta"],
    }
