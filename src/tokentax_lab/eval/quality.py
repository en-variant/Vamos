from __future__ import annotations

import numpy as np
import random


def bootstrap_accuracy(correct: list[bool], n_bootstrap: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = np.asarray(correct, dtype=float)
    if values.size == 0:
        raise ValueError("No evaluation examples")
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    return {"accuracy": float(values.mean()), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975))}


def group_bootstrap_accuracy(
    correct: list[bool],
    groups: list[str],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap passage IDs so sibling questions never act as independent samples."""
    if len(correct) != len(groups):
        raise ValueError("correct and groups must have the same length")
    if not correct:
        raise ValueError("No evaluation examples")
    grouped: dict[str, list[float]] = {}
    for group, value in zip(groups, correct, strict=True):
        grouped.setdefault(group, []).append(float(value))
    keys = list(grouped)
    if not keys:
        raise ValueError("No passage groups")
    rng = np.random.default_rng(seed)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    group_successes = np.asarray([sum(grouped[key]) for key in keys], dtype=float)
    group_sizes = np.asarray([len(grouped[key]) for key in keys], dtype=float)
    sampled_indices = rng.integers(0, len(keys), size=(n_bootstrap, len(keys)))
    samples = (
        group_successes[sampled_indices].sum(axis=1)
        / group_sizes[sampled_indices].sum(axis=1)
    )
    values = np.asarray(correct, dtype=float)
    return {
        "accuracy": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "groups": len(keys),
        "n": len(values),
        "interval_method": "passage_cluster_bootstrap",
    }


def compare_confidence_intervals(first: dict, second: dict) -> str:
    if first["ci95_high"] < second["ci95_low"] or second["ci95_high"] < first["ci95_low"]:
        return "measurable_difference"
    return "no measurable difference"


def chrF(hypotheses: list[str], references: list[str]) -> dict:
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("Install sacrebleu") from exc
    score = sacrebleu.corpus_chrf([r for r in references], [h for h in hypotheses])
    return {"chrf": float(score.score), "signature": score.signature}


def shuffle_options(options: list[str], seed: int) -> list[str]:
    out = list(options)
    random.Random(seed).shuffle(out)
    return out


def letter_bias_check(records: list[dict], answer_key: str = "answer") -> dict:
    counts = {}
    for r in records:
        letter = r.get(answer_key)
        counts[letter] = counts.get(letter, 0) + 1
    total = sum(counts.values())
    return {"option_counts": counts, "share": {k: v / total for k, v in counts.items()} if total else {}}


def deterministic_check(outputs_a: list[str], outputs_b: list[str]) -> dict:
    matches = sum(a == b for a, b in zip(outputs_a, outputs_b))
    n = min(len(outputs_a), len(outputs_b))
    return {"n": n, "greedy_match_rate": matches / n if n else 0.0, "deterministic": bool(n and matches == n)}
