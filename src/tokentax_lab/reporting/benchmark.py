from __future__ import annotations

import tiktoken


def reproduce_12x(
    language_texts: dict[str, list[str]],
    tokenizer_name: str,
    language: str = "kn",
) -> dict:
    if language not in language_texts or "en" not in language_texts:
        raise ValueError(f"Parallel {language} and English texts are required")
    if len(language_texts[language]) != len(language_texts["en"]):
        raise ValueError("Kannada and English samples must be parallel and equal-length")
    encoding = tiktoken.get_encoding(tokenizer_name)
    lang_counts = [len(encoding.encode(text)) for text in language_texts[language]]
    english_counts = [len(encoding.encode(text)) for text in language_texts["en"]]
    lang_total = sum(lang_counts)
    english_total = sum(english_counts)
    if english_total == 0:
        raise ValueError("English sample token count is zero")
    return {
        "claim_reference": "12x claim from the challenge booklet; source booklet not supplied",
        "language": language,
        "tokenizer": tokenizer_name,
        "sentences": len(lang_counts),
        "language_tokens": lang_total,
        "english_tokens": english_total,
        "measured_ratio_to_english": lang_total / english_total,
        "measurement_method": "aggregate tokenizer token count over the same parallel FLORES-200 devtest IDs",
        "note": "This is our reproducible calculation, not an assertion that the booklet's exact setup was reproduced.",
    }
