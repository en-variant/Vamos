from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import time
import tiktoken

from ..data import load_config
from ..optimizer.features import extract_features
from ..optimizer.router import Router
from ..optimizer.safety import SafetyGate
from ..optimizer.strategies import (
    MixedLanguageStrategy,
    NativeStrategy,
    RomanizeStrategy,
    UnicodeCleanupStrategy,
)
from ..audit.engine import default_tokenizer_specs


@lru_cache(maxsize=4)
def _load_hf_tokenizer(model_id: str):
    import os
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, use_fast=True, token=os.environ.get("HF_TOKEN")
    )


def _count_prompt_tokens(text: str, tokenizer_name: str) -> tuple[int | None, str | None, str | None]:
    try:
        encoding = tiktoken.get_encoding(tokenizer_name)
    except (KeyError, ValueError):
        model_id = default_tokenizer_specs().get(tokenizer_name)
        if model_id is None:
            return None, None, f"Unknown tokenizer: {tokenizer_name}"
        try:
            tokenizer = _load_hf_tokenizer(model_id)
        except Exception as exc:
            return None, model_id, f"{type(exc).__name__}: {exc}"
        return len(tokenizer.encode(text, add_special_tokens=False)), model_id, None
    return len(encoding.encode(text)), None, None


def token_receipt(
    text: str,
    strategy: str = "native",
    quality_risk: str = "unmeasured",
    latency_ms: float = 0.0,
    tokenizer_names: tuple[str, ...] = ("cl100k_base", "o200k_base"),
) -> dict:
    config = load_config()
    price = config["cost"]
    inr_per_million = (
        float(price["usd_per_million_input_tokens"]) * float(price["usd_to_inr"])
    )
    rows = []
    for name in tokenizer_names:
        count, model_id, reason = _count_prompt_tokens(text, name)
        rows.append({
            "tokenizer": name,
            "model_id": model_id,
            "status": "available" if count is not None else "SKIPPED",
            "reason": reason,
            "tokens": count,
            "estimated_cost_inr": (
                round(count / 1_000_000 * inr_per_million, 8)
                if count is not None else None
            ),
            "cost_basis": (
                f"Counterfactual using {price['model']} at USD "
                f"{price['usd_per_million_input_tokens']}/1M input tokens"
            ) if count is not None else "Unavailable: tokenizer could not be loaded.",
        })
    available_rows = [row for row in rows if row["tokens"] is not None]
    cheapest = min(available_rows, key=lambda row: row["estimated_cost_inr"]) if available_rows else None
    return {
        "text": text,
        "strategy": strategy,
        "quality_risk": quality_risk,
        "added_latency_ms": latency_ms,
        "tokenizers": rows,
        "lowest_counterfactual_cost_tokenizer": cheapest["tokenizer"] if cheapest else None,
        "price": {
            "model": price["model"],
            "tier": price["tier"],
            "usd_per_million_input_tokens": price["usd_per_million_input_tokens"],
            "price_source": price["price_source"],
            "price_recorded_date": price["price_recorded_date"],
            "price_valid_through": price["price_valid_through"],
            "usd_to_inr": price["usd_to_inr"],
            "fx_source": price["fx_source"],
            "fx_date": price["fx_date"],
        },
        "interpretation": (
            "These are counterfactual costs: cl100k/o200k counts are not Gemini's billed token count. "
            "Use the Gemini API countTokens endpoint for an exact Gemini estimate."
        ),
    }


def compare_tokenizers(
    texts: dict[str, str],
    tokenizer_names: tuple[str, ...],
) -> list[dict]:
    """Compare counts for named text variants; unavailable HF assets remain explicit."""
    rows = []
    for tokenizer_name in tokenizer_names:
        counts = {}
        model_id = None
        reason = None
        for label, text in texts.items():
            count, current_model_id, current_reason = _count_prompt_tokens(
                text, tokenizer_name
            )
            model_id = model_id or current_model_id
            if count is None:
                reason = current_reason
                break
            counts[label] = count
        row = {
            "tokenizer": tokenizer_name,
            "model_id": model_id,
            "status": "available" if len(counts) == len(texts) else "SKIPPED",
            "reason": reason,
        }
        row.update({f"{label}_tokens": counts.get(label) for label in texts})
        if len(counts) == len(texts):
            first_label, second_label = texts
            first_count = counts[first_label]
            second_count = counts[second_label]
            row.update({
                "tokens_saved": first_count - second_count,
                "token_change_percent": (
                    (second_count - first_count) / first_count * 100
                    if first_count else None
                ),
            })
        else:
            row.update({"tokens_saved": None, "token_change_percent": None})
        rows.append(row)
    return rows


def best_tokenizer(
    comparison_rows: list[dict],
    text_label: str = "English translation",
) -> dict | None:
    """Return the available tokenizer with the fewest pieces for the selected text."""
    count_key = f"{text_label}_tokens"
    candidates = [
        row for row in comparison_rows
        if row.get("status") == "available"
        and isinstance(row.get(count_key), int)
    ]
    if not candidates:
        return None
    minimum = min(row[count_key] for row in candidates)
    winners = [row for row in candidates if row[count_key] == minimum]
    return {
        "tokenizers": [row["tokenizer"] for row in winners],
        "tokens": minimum,
        "tied": len(winners) > 1,
        "text_label": text_label,
    }


def prompt_receipt(
    text: str,
    tokenizer_names: tuple[str, ...] = ("cl100k_base", "o200k_base"),
    results_dir: str | Path = "results",
) -> dict:
    results_dir = Path(results_dir)
    feature = extract_features(text)
    non_english_shares = [
        ("kn", feature["kannada_share"]),
        ("hi", feature["hindi_share"]),
        ("ta", feature["tamil_share"]),
    ]
    strongest_language, strongest_share = max(non_english_shares, key=lambda item: item[1])
    language = strongest_language if strongest_share > 0 else "en"
    router_path = results_dir / f"router_{language}.joblib"
    strategy_label = "native"
    if router_path.exists():
        router = Router()
        router.load(router_path)
        strategy_label = router.predict(text)
    selected_label = strategy_label
    strategy_id = strategy_label.lower()
    if strategy_id.startswith("romanize_"):
        strategy = RomanizeStrategy(strategy_id.removeprefix("romanize_"))
    elif strategy_id == "unicode_cleanup":
        strategy = UnicodeCleanupStrategy()
    elif strategy_id == "mixed_language":
        strategy = MixedLanguageStrategy()
    elif strategy_id == "pivot":
        strategy = NativeStrategy()
        strategy_label = "native_fallback"
    else:
        strategy = NativeStrategy()
        strategy_label = "native"

    started = time.perf_counter()
    if strategy_id == "pivot":
        rewritten, metadata = text, {
            "fallback": True,
            "reason": "Local pivot model is not preloaded in the receipt UI.",
            "quality_risk": "unmeasured",
        }
        accepted, gate_reason = False, metadata["reason"]
    else:
        rewritten, metadata = strategy.transform(text, language)
        accepted, gate_reason = SafetyGate().check(text, rewritten, metadata)
    transform_latency = (time.perf_counter() - started) * 1000
    fallback = not accepted
    if fallback:
        rewritten = text
        strategy_label = "native_fallback"
    receipt = token_receipt(
        rewritten,
        strategy=strategy_label,
        quality_risk=metadata.get("quality_risk", "unmeasured"),
        latency_ms=transform_latency,
        tokenizer_names=tokenizer_names,
    )
    for row in receipt["tokenizers"]:
        if row["status"] != "available":
            row["original_tokens"] = None
            row["tokens_saved"] = None
            continue
        original_count, _, reason = _count_prompt_tokens(text, row["tokenizer"])
        if original_count is None:
            row["status"] = "SKIPPED"
            row["reason"] = reason
            row["tokens"] = None
            row["estimated_cost_inr"] = None
            row["original_tokens"] = None
            row["tokens_saved"] = None
            continue
        row["original_tokens"] = original_count
        row["tokens_saved"] = original_count - row["tokens"]
    receipt.update({
        "language": language,
        "rewrite_accepted": accepted,
        "fallback_reason": gate_reason if fallback else None,
        "router_reason": (
            f"Trained passage-split router selected {selected_label}."
            if router_path.exists()
            else "No trained router artifact; native baseline selected."
        ),
    })
    quality_file = results_dir / f"evaluation_{language}.json"
    if quality_file.exists():
        measured = json.loads(quality_file.read_text(encoding="utf-8"))
        points = measured.get("summary", {}).get("points", [])
        target = next((point for point in points if point["name"] == strategy_label), None)
        native = next((point for point in points if point["name"] == "native"), None)
        if target and native:
            overlap = not (
                target["ci95_high"] < native["ci95_low"]
                or native["ci95_high"] < target["ci95_low"]
            )
            receipt["quality_risk"] = (
                "no measurable difference" if overlap else "measured accuracy difference"
            )
            receipt["quality_evidence"] = {
                "strategy_accuracy": target["accuracy"],
                "strategy_ci95": [target["ci95_low"], target["ci95_high"]],
                "native_accuracy": native["accuracy"],
                "native_ci95": [native["ci95_low"], native["ci95_high"]],
                "n": target["n"],
            }
    return receipt


def improvement_ledger(rows: list[dict]) -> list[dict]:
    return [{
        "improvement": row.get("improvement", "unknown"),
        "fertility_drop": row.get("fertility_drop", "not measured"),
        "quality_delta": row.get("quality_delta", "not measured"),
        "latency_ms": row.get("latency_ms", "not measured"),
        "gpu_hours": row.get("gpu_hours", "not measured"),
        "extra_parameters": row.get("extra_parameters", "not measured"),
    } for row in rows]
