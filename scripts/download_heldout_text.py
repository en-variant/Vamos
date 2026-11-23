import argparse
import json
from pathlib import Path

from datasets import load_dataset


LANGS = {"kn": "20231101.kn", "hi": "20231101.hi", "ta": "20231101.ta", "en": "20231101.en"}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch separate Wikimedia Wikipedia train/eval text; never reads FLORES."
    )
    parser.add_argument("--output", default="data/heldout")
    parser.add_argument("--train-docs", type=int, default=5000)
    parser.add_argument("--eval-docs", type=int, default=500)
    args = parser.parse_args()
    if args.train_docs < 1 or args.eval_docs < 1:
        parser.error("--train-docs and --eval-docs must be positive")

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": "wikimedia/wikipedia",
        "configs": LANGS,
        "split": "train",
        "partition": {
            "train": f"first {args.train_docs} documents",
            "eval": f"next {args.eval_docs} documents; disjoint document slices",
        },
        "use": "Tokenizer adaptation and bits-per-byte evaluation only; FLORES is not used for training.",
        "source_card": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "license_note": "Check the dataset card and underlying Wikimedia article licenses before redistribution.",
    }
    for language, config in LANGS.items():
        dataset = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
        train_lines, eval_lines = [], []
        for index, row in enumerate(dataset):
            text = " ".join(str(row.get("text", "")).split())
            if not text:
                continue
            if len(train_lines) < args.train_docs:
                train_lines.append(text)
            elif len(eval_lines) < args.eval_docs:
                eval_lines.append(text)
            if len(train_lines) == args.train_docs and len(eval_lines) == args.eval_docs:
                break
        if len(train_lines) != args.train_docs or len(eval_lines) != args.eval_docs:
            raise RuntimeError(
                f"{language} stream ended early: train={len(train_lines)}, eval={len(eval_lines)}"
            )
        language_dir = root / language
        language_dir.mkdir(parents=True, exist_ok=True)
        (language_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        (language_dir / "eval.txt").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
        provenance.setdefault("documents_written", {})[language] = {
            "train": len(train_lines),
            "eval": len(eval_lines),
        }
    (root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
