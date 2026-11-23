from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .audit.engine import AuditEngine, default_tokenizer_specs
from .audit.flores import (
    download_belebele,
    load_from_huggingface,
    load_local_flores,
)
from .data import load_config, save_json
from .eval.benchmark import (
    english_model_calibration,
    run_evaluation_diagnostics,
    run_strategy_suite,
    summarize_strategy_suite,
    train_router_and_compare,
    write_report,
)
from .eval.belebele import (
    BelebeleEvaluator,
    GeminiGenerator,
    LocalCausalGenerator,
    OpenAICompatibleGenerator,
)
from .reporting.receipts import token_receipt
from .reporting.benchmark import reproduce_12x
from .failure_lab import examples


def _price_engine() -> AuditEngine:
    config = load_config()
    price = config["cost"]
    return AuditEngine(
        context_tokens=config["context_tokens"],
        price_model=price["model"],
        price_usd_per_million=price["usd_per_million_input_tokens"],
        usd_to_inr=price["usd_to_inr"],
        price_recorded_date=price["price_recorded_date"],
        fx_date=price["fx_date"],
    )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="tokentax", description="TokenTax Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit FLORES and available tokenizer assets")
    audit.add_argument("--flores", default="data/flores")
    audit.add_argument("--download-flores", action="store_true")
    audit.add_argument("--output", default="results/audit")
    audit.add_argument("--skip-hf-tokenizers", action="store_true")

    download = sub.add_parser("download", help="Download gated FLORES and/or public Belebele")
    download.add_argument("--flores", action="store_true")
    download.add_argument("--belebele", action="store_true")

    receipt = sub.add_parser("receipt")
    receipt.add_argument("--text", required=True)

    evaluate = sub.add_parser("evaluate", help="Run the same-model Belebele strategy evaluation")
    evaluate.add_argument("--language", choices=("kn", "hi", "ta", "en"), default="kn")
    evaluate.add_argument(
        "--backend", choices=("openai", "gemini", "local"), default="local"
    )
    evaluate.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--limit", type=int, default=0, help="Items to evaluate; default 0 means the full split")
    evaluate.add_argument("--quality-floor", type=float, default=0.25)
    evaluate.add_argument("--budget-cap-inr", type=float, default=50.0)
    evaluate.add_argument(
        "--allow-paid", action="store_true",
        help="Explicitly enable paid OpenAI-compatible or Gemini API calls",
    )
    evaluate.add_argument("--skip-diagnostics", action="store_true")
    evaluate.add_argument("--diagnostic-items", type=int, default=20)

    sub.add_parser("failure-lab")
    sub.add_parser("status")
    args = parser.parse_args()

    if args.command == "audit":
        flores_path = Path(args.flores)
        if args.download_flores and not (flores_path / "kn.devtest").exists():
            load_from_huggingface(flores_path)
        texts = load_local_flores(flores_path, ("kn", "hi", "ta", "en"))
        engine = _price_engine()
        rows = engine.audit_many(("cl100k_base", "o200k_base"), texts)
        claim_reproductions = [
            reproduce_12x(texts, encoding, "kn")
            for encoding in ("cl100k_base", "o200k_base")
        ]
        if not args.skip_hf_tokenizers:
            for name, model_id in default_tokenizer_specs().items():
                rows.extend(engine.audit_hf(model_id, texts, name))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
        save_json(output.with_suffix(".json"), {
            "dataset": "FLORES-200 devtest",
            "sentences_per_language": 1012,
            "cost_interpretation": "Counterfactual Gemini input rate for non-Gemini tokenizer counts",
            "price_source": load_config()["cost"]["price_source"],
            "fx_source": load_config()["cost"]["fx_source"],
            "12x_claim_reproduction": claim_reproductions,
            "rows": rows,
        })
        print(json.dumps({"csv": str(output.with_suffix(".csv")), "json": str(output.with_suffix(".json")), "rows": len(rows)}, indent=2))
    elif args.command == "download":
        if not args.flores and not args.belebele:
            parser.error("Choose at least one of --flores or --belebele")
        status = {}
        if args.flores:
            loaded = load_from_huggingface()
            status["flores_sentences_per_language"] = {key: len(value) for key, value in loaded.items()}
        if args.belebele:
            loaded = download_belebele()
            status["belebele_questions_per_language"] = {key: len(value) for key, value in loaded.items()}
        print(json.dumps(status, ensure_ascii=False, indent=2))
    elif args.command == "receipt":
        print(json.dumps(token_receipt(args.text), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        if args.limit < 0:
            parser.error("--limit cannot be negative")
        if args.backend == "openai":
            if not args.allow_paid:
                parser.error("API calls may cost money; pass --allow-paid and set a budget cap explicitly.")
            generator = OpenAICompatibleGenerator(budget_cap_inr=args.budget_cap_inr)
        elif args.backend == "gemini":
            if not args.allow_paid:
                parser.error("Gemini calls cost money; pass --allow-paid and set a budget cap explicitly.")
            generator = GeminiGenerator(budget_cap_inr=args.budget_cap_inr)
        else:
            generator = LocalCausalGenerator(args.model, args.device)
        all_items = download_belebele()
        evaluator = BelebeleEvaluator(generator)
        try:
            calibration = english_model_calibration(evaluator, all_items["en"])
        except RuntimeError as exc:
            if "budget limit reached" not in str(exc).lower():
                raise
            report_path = Path(f"results/evaluation_{args.language}.json")
            write_report(report_path, {
                "status": "blocked_english_calibration_budget",
                "reason": str(exc),
                "spent_inr": getattr(generator, "spent_inr", 0.0),
            })
            print(json.dumps({"status": "blocked_english_calibration_budget", "report": str(report_path)}, indent=2))
            return
        if not calibration["passed"] and args.backend == "local":
            fallback_model = "Qwen/Qwen2.5-1.5B-Instruct"
            generator = LocalCausalGenerator(fallback_model, args.device)
            evaluator = BelebeleEvaluator(generator)
            calibration = english_model_calibration(evaluator, all_items["en"])
            calibration["fallback_model_used"] = fallback_model
        if not calibration["passed"]:
            report_path = Path(f"results/evaluation_{args.language}.json")
            write_report(report_path, {
                "status": "blocked_model_below_english_chance_floor",
                "calibration": calibration,
                "message": "English calibration 95% CI lower bound did not exceed 25%; no strategy results reported.",
            })
            print(json.dumps({"status": "blocked_model_below_english_chance_floor", "report": str(report_path)}, indent=2))
            return
        calibration_passages = set(calibration["calibration_passages"])
        evaluation_items = [
            item for item in all_items[args.language]
            if item.get("link") not in calibration_passages
        ]
        items = evaluation_items
        limit = args.limit or None
        suite = run_strategy_suite(evaluator, items, args.language, max_items=limit)
        summary = summarize_strategy_suite(suite)
        report = {"calibration": calibration, "suite": suite, "summary": summary}
        if args.skip_diagnostics:
            report["diagnostics"] = {"status": "not_run", "reason": "disabled by --skip-diagnostics"}
        else:
            try:
                report["diagnostics"] = run_evaluation_diagnostics(
                    evaluator, items, args.language, sample_size=args.diagnostic_items,
                )
            except RuntimeError as exc:
                if "budget limit reached" not in str(exc).lower():
                    raise
                report["diagnostics"] = {
                    "status": "budget_limited",
                    "reason": str(exc),
                }
        try:
            router = train_router_and_compare(
                suite, items[:limit] if limit else items,
                args.language, quality_floor=args.quality_floor,
                model_path=Path(f"results/router_{args.language}.joblib"),
            )
        except ValueError as exc:
            report["router"] = {"status": "blocked", "reason": str(exc)}
        else:
            report["router"] = router
            for name, result in router["test_comparison"].items():
                report["summary"]["points"].append({
                    "name": f"{name}_holdout",
                    "tokens": result["mean_tokens"],
                    "accuracy": result["accuracy"],
                    "ci95_low": result["ci95_low"],
                    "ci95_high": result["ci95_high"],
                    "latency_ms": result["mean_latency_ms"],
                    "cost_inr": result["cost_inr"],
                    "n": result["n"],
                    "groups": result["groups"],
                    "evidence_status": "held-out measured",
                })
        output = Path(f"results/evaluation_{args.language}.json")
        write_report(output, report)
        print(json.dumps({
            "report": str(output),
            "status": summary["status"],
            "complete_items": summary.get("complete_items", 0),
            "router_status": report["router"].get("evidence_status", report["router"].get("status")),
            "spent_inr": getattr(generator, "spent_inr", 0.0),
        }, ensure_ascii=False, indent=2))
    elif args.command == "failure-lab":
        print(json.dumps(examples(), ensure_ascii=False, indent=2))
    elif args.command == "status":
        config = load_config()
        status = {
            "project": "TokenTax Lab",
            "datasets": {
                "FLORES-200": "gated; requires accepted terms and HF_TOKEN",
                "Belebele": "downloadable from public Hugging Face files",
            },
            "credentials_configured": {
                "HF_TOKEN": bool(__import__("os").environ.get("HF_TOKEN")),
                "GOOGLE_API_KEY": bool(__import__("os").environ.get("GOOGLE_API_KEY")),
                "OPENAI_API_KEY": bool(
                    __import__("os").environ.get("OPENAI_API_KEY")
                    or __import__("os").environ.get("OPENAPI_API_KEY")
                ),
            },
            "pricing": {
                "model": config["cost"]["model"],
                "rate": config["cost"]["usd_per_million_input_tokens"],
                "price_recorded_date": config["cost"]["price_recorded_date"],
                "fx_usd_to_inr": config["cost"]["usd_to_inr"],
                "fx_date": config["cost"]["fx_date"],
            },
        }
        print(json.dumps(status, ensure_ascii=False, indent=2))
