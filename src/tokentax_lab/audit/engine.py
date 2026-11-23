from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import tiktoken


@dataclass
class AuditResult:
    tokenizer: str
    language: str
    sentences: int
    total_tokens: int
    tokens_per_sentence: float
    ratio_to_english: float | None
    fertility_words: float
    bytes_per_token: float
    effective_context_sentences: float
    estimated_cost_inr: float | None
    cost_basis: str
    price_model: str | None
    price_usd_per_million: float | None
    fx_usd_to_inr: float | None
    price_recorded_date: str | None
    fx_date: str | None


class AuditEngine:
    """Compare tokenizer segmentation; price is an explicitly labeled counterfactual."""

    def __init__(
        self,
        context_tokens: int = 8192,
        price_model: str | None = None,
        price_usd_per_million: float | None = None,
        usd_to_inr: float | None = None,
        price_recorded_date: str | None = None,
        fx_date: str | None = None,
    ):
        self.context_tokens = context_tokens
        self.price_model = price_model
        self.price_usd_per_million = price_usd_per_million
        self.usd_to_inr = usd_to_inr
        self.price_recorded_date = price_recorded_date
        self.fx_date = fx_date

    def count_tiktoken(self, encoding_name: str, texts: Iterable[str]) -> tuple[int, list[int]]:
        enc = tiktoken.get_encoding(encoding_name)
        counts = [len(enc.encode(text)) for text in texts]
        return sum(counts), counts

    def _row(self, tokenizer_name: str, language: str, texts: list[str], counts: list[int]) -> AuditResult:
        n = len(texts)
        total = sum(counts)
        words = sum(max(1, len(text.split())) for text in texts)
        byte_count = sum(len(text.encode("utf-8")) for text in texts)
        rate_available = (
            self.price_usd_per_million is not None
            and self.usd_to_inr is not None
        )
        cost = (
            total / 1_000_000 * self.price_usd_per_million * self.usd_to_inr
            if rate_available
            else None
        )
        return AuditResult(
            tokenizer=tokenizer_name,
            language=language,
            sentences=n,
            total_tokens=total,
            tokens_per_sentence=total / n if n else 0.0,
            ratio_to_english=None,
            fertility_words=total / words if words else 0.0,
            bytes_per_token=byte_count / total if total else 0.0,
            effective_context_sentences=self._effective_context_sentences(counts),
            estimated_cost_inr=cost,
            cost_basis=(
                f"Counterfactual {self.price_model} published input rate applied to "
                f"{tokenizer_name} token count; not an actual Gemini billing token count."
                if rate_available else "Unpriced: configure a dated model price and FX rate."
            ),
            price_model=self.price_model if rate_available else None,
            price_usd_per_million=self.price_usd_per_million if rate_available else None,
            fx_usd_to_inr=self.usd_to_inr if rate_available else None,
            price_recorded_date=self.price_recorded_date if rate_available else None,
            fx_date=self.fx_date if rate_available else None,
        )

    def _effective_context_sentences(self, counts: list[int]) -> float:
        if not counts:
            return 0.0
        end = 0
        running_tokens = 0
        total_sentences = 0
        for start in range(len(counts)):
            while end < len(counts) and running_tokens + counts[end] <= self.context_tokens:
                running_tokens += counts[end]
                end += 1
            total_sentences += end - start
            if end == start:
                end += 1
            else:
                running_tokens -= counts[start]
        return total_sentences / len(counts)

    def audit_tiktoken(self, encoding_name: str, language: str, texts: Iterable[str]) -> AuditResult:
        texts = list(texts)
        _, counts = self.count_tiktoken(encoding_name, texts)
        return self._row(encoding_name, language, texts, counts)

    def audit_many(self, encoding_names: Iterable[str], language_texts: dict[str, list[str]]) -> list[dict]:
        rows = []
        for encoding_name in encoding_names:
            encoding_rows = {
                lang: self.audit_tiktoken(encoding_name, lang, texts)
                for lang, texts in language_texts.items()
            }
            english = encoding_rows.get("en")
            for language, result in encoding_rows.items():
                row = asdict(result)
                if english and english.tokens_per_sentence:
                    row["ratio_to_english"] = result.tokens_per_sentence / english.tokens_per_sentence
                rows.append(row)
        return rows

    def audit_hf(
        self,
        model_id: str,
        language_texts: dict[str, list[str]],
        friendly_name: str | None = None,
    ) -> list[dict]:
        name = friendly_name or model_id
        try:
            from transformers import AutoTokenizer
            import os
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True, token=os.environ.get("HF_TOKEN")
            )
        except Exception as exc:
            return [{
                "tokenizer": name,
                "model_id": model_id,
                "status": "SKIPPED",
                "reason": f"{type(exc).__name__}: {exc}",
            }]
        rows = []
        for lang, texts in language_texts.items():
            counts = [
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in texts
            ]
            result = self._row(name, lang, texts, counts)
            rows.append(asdict(result) | {"model_id": model_id, "status": "available"})
        english = next(
            (row["tokens_per_sentence"] for row in rows if row["language"] == "en"),
            None,
        )
        if english:
            for row in rows:
                row["ratio_to_english"] = row["tokens_per_sentence"] / english
        return rows


def default_tokenizer_specs() -> dict[str, str]:
    return {
        "Qwen2.5": "Qwen/Qwen2.5-0.5B",
        "Llama 3": "meta-llama/Meta-Llama-3-8B",
        "Gemma": "google/gemma-2-2b",
        "Sarvam-1": "sarvamai/sarvam-1",
    }
