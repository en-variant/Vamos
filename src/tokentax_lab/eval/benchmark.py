from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import random

from sklearn.model_selection import GroupShuffleSplit

from ..optimizer.router import Router
from ..optimizer.strategies import (
    MixedLanguageStrategy,
    NativeStrategy,
    PivotStrategy,
    RomanizeStrategy,
    UnicodeCleanupStrategy,
)
from ..optimizer.break_even import find_break_even
from .quality import compare_confidence_intervals, group_bootstrap_accuracy
from .quality import letter_bias_check, deterministic_check


def default_strategies() -> list:
    return [
        NativeStrategy(),
        UnicodeCleanupStrategy(),
        MixedLanguageStrategy(),
        RomanizeStrategy("itrans"),
        RomanizeStrategy("iso"),
        RomanizeStrategy("casual"),
        PivotStrategy(),
    ]


def _item_id(item: dict, language: str) -> str:
    return f"{language}|{item.get('link', '')}|{item.get('question_number', '')}"


def english_model_calibration(
    evaluator,
    english_items: list[dict],
    sample_size: int = 100,
    chance_level: float = 0.25,
    seed: int = 42,
) -> dict:
    """Require the passage-bootstrap CI lower bound to exceed multiple-choice chance."""
    by_passage = {}
    for item in english_items:
        by_passage.setdefault(item.get("link", str(item.get("question_number", ""))), item)
    candidates = list(by_passage.values())
    random.Random(seed).shuffle(candidates)
    selected = candidates[:sample_size]
    if len(selected) < 20:
        raise ValueError(f"English calibration needs at least 20 unique passages; found {len(selected)}")
    rows = [evaluator.predict(item, language="en") for item in selected]
    interval = group_bootstrap_accuracy(
        [row["correct"] for row in rows],
        [row["passage_id"] for row in rows],
    )
    interval["chance_level"] = chance_level
    interval["passed"] = interval["ci95_low"] > chance_level
    return {
        **interval,
        "model": rows[0]["model"],
        "calibration_passages": [row["passage_id"] for row in rows],
        "n": len(rows),
    }


def run_evaluation_diagnostics(
    evaluator,
    items: list[dict],
    language: str,
    sample_size: int = 20,
    seed: int = 42,
) -> dict:
    candidates = items[:sample_size]
    if not candidates:
        return {"status": "blocked", "reason": "No items available for diagnostics"}
    shuffled_rows = []
    repeated_a, repeated_b = [], []
    for index, item in enumerate(candidates):
        order = [1, 2, 3, 4]
        random.Random(seed + index).shuffle(order)
        shuffled_rows.append(evaluator.predict(
            item, language=language, option_order=order
        ))
        first = evaluator.predict(item, language=language)
        second = evaluator.predict(item, language=language)
        repeated_a.append(first["raw_output"])
        repeated_b.append(second["raw_output"])
    distribution = letter_bias_check(
        [{"answer": row["prediction"]} for row in shuffled_rows]
    )
    return {
        "status": "measured",
        "n": len(candidates),
        "option_shuffle_seed": seed,
        "shuffled_prediction_distribution": distribution,
        "shuffled_accuracy": sum(row["correct"] for row in shuffled_rows) / len(shuffled_rows),
        "greedy_determinism": deterministic_check(repeated_a, repeated_b),
    }


def run_strategy_suite(
    evaluator,
    items: list[dict],
    language: str,
    strategies: list | None = None,
    max_items: int | None = None,
) -> dict:
    strategies = strategies or default_strategies()
    selected_items = items[:max_items] if max_items else items
    results: list[dict] = []
    status = "complete"
    budget_error = None
    for item in selected_items:
        for strategy in strategies:
            try:
                row = evaluator.predict(item, strategy=strategy, language=language)
            except RuntimeError as exc:
                if "budget limit reached" not in str(exc).lower():
                    raise
                status = "budget_limited"
                budget_error = str(exc)
                break
            row["item_id"] = _item_id(item, language)
            results.append(row)
        if status != "complete":
            break
    return {
        "status": status,
        "language": language,
        "strategies": [strategy.name + (f"_{strategy.scheme}" if isinstance(strategy, RomanizeStrategy) else "") for strategy in strategies],
        "results": results,
        "budget_stop_reason": budget_error,
    }


def summarize_strategy_suite(suite: dict) -> dict:
    grouped = defaultdict(dict)
    for row in suite["results"]:
        grouped[row["item_id"]][row["strategy"] + (f"_{row['scheme']}" if row.get("scheme") else "")] = row
    if not grouped:
        return {"status": suite["status"], "points": [], "comparisons": {}}
    complete_items = {
        item_id: rows for item_id, rows in grouped.items()
        if all(name in rows for name in suite["strategies"])
    }
    points = []
    for strategy in suite["strategies"]:
        rows = [items[strategy] for items in complete_items.values()]
        if not rows:
            continue
        interval = group_bootstrap_accuracy(
            [row["correct"] for row in rows],
            [row["passage_id"] for row in rows],
        )
        points.append({
            "name": strategy,
            "tokens": sum(row.get("prompt_tokens", 0) for row in rows) / len(rows),
            "accuracy": interval["accuracy"],
            "ci95_low": interval["ci95_low"],
            "ci95_high": interval["ci95_high"],
            "latency_ms": sum(row.get("end_to_end_latency_ms", row.get("latency_ms", 0.0)) for row in rows) / len(rows),
            "cost_inr": sum(row.get("cost_inr", 0.0) for row in rows),
            "n": len(rows),
            "groups": interval["groups"],
            "evidence_status": "measured",
        })
    baseline = next((point for point in points if point["name"] == "native"), None)
    comparisons = {}
    if baseline:
        for point in points:
            comparisons[point["name"]] = compare_confidence_intervals(baseline, point)
    pivot = next((point for point in points if point["name"] == "pivot"), None)
    break_even = None
    if baseline and pivot:
        saved_fraction = (
            (baseline["tokens"] - pivot["tokens"]) / baseline["tokens"]
            if baseline["tokens"] else 0.0
        )
        break_even = find_break_even(
            native_latency_ms=baseline["latency_ms"],
            translation_latency_ms=max(0.0, pivot["latency_ms"] - baseline["latency_ms"]),
            prompt_threshold_tokens=baseline["tokens"],
            saved_fraction=saved_fraction,
        )
    return {
        "status": suite["status"],
        "language": suite["language"],
        "complete_items": len(complete_items),
        "points": points,
        "comparisons_to_native": comparisons,
        "pivot_latency_break_even": break_even,
        "excluded_partial_items": len(grouped) - len(complete_items),
        "budget_stop_reason": suite.get("budget_stop_reason"),
    }


def _summarize_selected(rows: list[dict], name: str) -> dict:
    ci = group_bootstrap_accuracy(
        [row["correct"] for row in rows],
        [row["passage_id"] for row in rows],
    )
    return {
        "name": name,
        **ci,
        "mean_tokens": sum(row.get("prompt_tokens", 0) for row in rows) / len(rows),
        "cost_inr": sum(row.get("cost_inr", 0.0) for row in rows),
        "mean_latency_ms": sum(
            row.get("end_to_end_latency_ms", row.get("latency_ms", 0.0))
            for row in rows
        ) / len(rows),
    }


def train_router_and_compare(
    suite: dict,
    items: list[dict],
    language: str,
    quality_floor: float = 0.25,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str | Path | None = None,
) -> dict:
    """Train/score by passage-group split; router labels use training passages only."""
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in suite["results"]:
        key = row["item_id"]
        by_id[key][row["strategy"] + (f"_{row['scheme']}" if row.get("scheme") else "")] = row
    item_by_id = {_item_id(item, language): item for item in items}
    common = [key for key, rows in by_id.items() if key in item_by_id and "native" in rows]
    names = suite["strategies"]
    common = [key for key in common if all(name in by_id[key] for name in names)]
    if len(common) < 5:
        raise ValueError("Need at least five fully evaluated items to split by passage")
    group_values = [by_id[key]["native"]["passage_id"] for key in common]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(common, groups=group_values))
    train_ids = [common[i] for i in train_indices]
    test_ids = [common[i] for i in test_indices]

    measured_train_accuracy = {
        name: sum(by_id[key][name]["correct"] for key in train_ids) / len(train_ids)
        for name in names
    }
    eligible = [name for name in names if measured_train_accuracy[name] >= quality_floor]
    if "native" not in eligible:
        eligible.append("native")

    texts, labels = [], []
    for key in train_ids:
        item = item_by_id[key]
        texts.append(item["flores_passage"] + "\n" + item["question"])
        labels.append(min(eligible, key=lambda n: by_id[key][n].get("prompt_tokens", float("inf"))))
    router = Router(max_depth=4, random_state=random_state).fit(texts, labels)
    if model_path:
        router.save(model_path)

    routed_rows, oracle_rows = [], []
    for key in test_ids:
        item = item_by_id[key]
        text = item["flores_passage"] + "\n" + item["question"]
        selected = router.predict(text)
        routed_rows.append(by_id[key][selected])
        correct_options = [by_id[key][name] for name in names if by_id[key][name]["correct"]]
        oracle_rows.append(
            min(correct_options, key=lambda row: row.get("prompt_tokens", float("inf")))
            if correct_options else by_id[key]["native"]
        )
    test_results = {"router": _summarize_selected(routed_rows, "router")}
    test_results["oracle"] = _summarize_selected(oracle_rows, "oracle")
    for name in ("native", "romanize_itrans", "pivot"):
        if name in names:
            test_results[f"always_{name}"] = _summarize_selected(
                [by_id[key][name] for key in test_ids], f"always_{name}"
            )
    return {
        "split": "GroupShuffleSplit by passage_id",
        "train_passages": sorted({by_id[key]["native"]["passage_id"] for key in train_ids}),
        "test_passages": sorted({by_id[key]["native"]["passage_id"] for key in test_ids}),
        "train_items": len(train_ids),
        "test_items": len(test_ids),
        "quality_floor": quality_floor,
        "train_strategy_accuracy": measured_train_accuracy,
        "eligible_training_strategies": eligible,
        "test_comparison": test_results,
        "router_depth": router.max_depth,
        "router_model": str(model_path) if model_path else None,
        "evidence_status": "measured",
    }


def write_report(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
