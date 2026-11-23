import argparse
import json
import time
from pathlib import Path

from tokentax_lab.audit.flores import load_local_flores
from tokentax_lab.optimizer.strategies import PivotStrategy
from tokentax_lab.eval.quality import chrF


def main():
    parser = argparse.ArgumentParser(
        description="Measure local NLLB Kannada/Hindi/Tamil-to-English translation with chrF."
    )
    parser.add_argument("--flores", default="data/flores")
    parser.add_argument("--language", choices=("kn", "hi", "ta"), default="kn")
    parser.add_argument("--limit", type=int, default=1012)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output", default="results/translation")
    args = parser.parse_args()
    if not 1 <= args.limit <= 1012:
        parser.error("--limit must be between 1 and 1,012")
    texts = load_local_flores(args.flores, ("kn", "hi", "ta", "en"))
    strategy = PivotStrategy(model_id=args.model, device="cpu")
    hypotheses, latency = [], []
    for text in texts[args.language][:args.limit]:
        hypothesis, metadata = strategy.transform(text, args.language)
        if metadata.get("fallback"):
            raise RuntimeError(f"Local translation failed: {metadata.get('reason')}")
        hypotheses.append(hypothesis)
        latency.append(metadata["added_latency_ms"])
    references = texts["en"][:len(hypotheses)]
    score = chrF(hypotheses, references)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "FLORES-200 devtest",
        "language": args.language,
        "model": args.model,
        "sentences": len(hypotheses),
        "chrf": score,
        "mean_translation_latency_ms": sum(latency) / len(latency),
        "total_translation_latency_seconds": sum(latency) / 1000,
        "wall_clock_seconds": sum(latency) / 1000,
        "hypotheses": hypotheses,
        "references": references,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / f"translation_{args.language}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key not in {"hypotheses", "references"}}, indent=2))


if __name__ == "__main__":
    main()
