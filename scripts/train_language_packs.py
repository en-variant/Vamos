import argparse
from pathlib import Path

from tokentax_lab.tokenizer.packs import LanguagePackTrainer, merge_language_packs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--text-dir", default="data/heldout")
    parser.add_argument("--packs-dir", default="results/packs")
    parser.add_argument("--output", default="results/extended-tokenizer")
    args = parser.parse_args()

    trainer = LanguagePackTrainer(args.vocab_size)
    pack_files = []
    for language in ("kn", "hi", "ta"):
        text = Path(args.text_dir) / language / "train.txt"
        pack = trainer.train(text, args.packs_dir, language)
        pack_files.append(pack["tokenizer_json"])
    result = merge_language_packs(args.base, pack_files, args.output)
    print(f"Merged {result['added_token_count']} added tokens into {result['output_dir']}")


if __name__ == "__main__":
    main()
