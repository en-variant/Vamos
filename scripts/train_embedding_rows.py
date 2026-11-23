import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokentax_lab.eval.bpb import corpus_bpb
from tokentax_lab.tokenizer.adaptation import add_lora, freeze_backbone
from tokentax_lab.tokenizer.packs import MeanEmbeddingInitializer


def load_text(path: Path, limit: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Required text file missing: {path}")
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts[:limit] if limit else texts


def fertility(tokenizer, texts: list[str]) -> float:
    token_count = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
    word_count = sum(max(1, len(text.split())) for text in texts)
    return token_count / word_count


def generation_tokens_per_second(model, tokenizer, text: str, device: str) -> float:
    inputs = tokenizer(text[:600], return_tensors="pt", truncation=True, max_length=256).to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
    return output_tokens / max(elapsed, 1e-9)


def main():
    parser = argparse.ArgumentParser(
        description="Lightly adapt only added embedding/output rows and measure bits per byte."
    )
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokenizer", default="results/extended-tokenizer")
    parser.add_argument("--manifest", default="results/extended-tokenizer/language_pack_manifest.json")
    parser.add_argument("--text-dir", default="data/heldout")
    parser.add_argument("--output", default="results/adapted-mean")
    parser.add_argument("--init", choices=("mean", "random"), default="mean")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-docs", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_length < 2:
        parser.error("--epochs must be positive and --max-length at least 2")

    text_dir = Path(args.text_dir)
    train_texts = []
    for language in ("kn", "hi", "ta"):
        train_texts.extend(load_text(text_dir / language / "train.txt", args.max_train_docs))
    indic_eval = [
        text for language in ("kn", "hi", "ta")
        for text in load_text(text_dir / language / "eval.txt", 0)
    ]
    english_eval = load_text(text_dir / "en" / "eval.txt", 0)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    new_tokens = manifest["new_tokens"]
    new_token_ids = [
        token_id for token, token_id in AutoTokenizer.from_pretrained(args.tokenizer).get_vocab().items()
        if token in set(new_tokens)
    ]
    new_token_ids = sorted(set(new_token_ids))
    if not new_token_ids:
        raise ValueError("No added token IDs found in the tokenizer manifest")

    base_tokenizer = AutoTokenizer.from_pretrained(args.base)
    extended_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    baseline_model = AutoModelForCausalLM.from_pretrained(args.base).to(args.device)
    base_indic_bpb = corpus_bpb(baseline_model, base_tokenizer, indic_eval, args.device)
    base_english_bpb = corpus_bpb(baseline_model, base_tokenizer, english_eval, args.device)
    base_indic_fertility = fertility(base_tokenizer, indic_eval)
    baseline_speed = generation_tokens_per_second(
        baseline_model, base_tokenizer, indic_eval[0], args.device
    )
    del baseline_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(args.base)
    original_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.resize_token_embeddings(len(extended_tokenizer))
    initialization = MeanEmbeddingInitializer().initialize(
        model, base_tokenizer, extended_tokenizer, new_tokens,
        method=args.init,
    )
    if args.lora:
        model = add_lora(model)
    trainable_parameters = freeze_backbone(model, new_token_ids, train_lora=args.lora)
    model.to(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    started = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        for text in train_texts:
            batch = extended_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=args.max_length
            ).to(args.device)
            if batch["input_ids"].shape[1] < 2:
                continue
            loss = model(**batch, labels=batch["input_ids"]).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started

    model.eval()
    indic_bpb = corpus_bpb(model, extended_tokenizer, indic_eval, args.device)
    english_bpb = corpus_bpb(model, extended_tokenizer, english_eval, args.device)
    extended_fertility = fertility(extended_tokenizer, indic_eval)
    speed = generation_tokens_per_second(
        model, extended_tokenizer, indic_eval[0], args.device
    )
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    extended_tokenizer.save_pretrained(output_path)
    metrics = {
        "base_model": args.base,
        "initialization": args.init,
        "lora_enabled": args.lora,
        "device": args.device,
        "heldout_text_source": "Wikimedia Wikipedia train/eval document slices; never FLORES",
        "train_documents": len(train_texts),
        "indic_eval_documents": len(indic_eval),
        "english_eval_documents": len(english_eval),
        "epochs": args.epochs,
        "initialization_result": initialization,
        "extra_parameters": sum(parameter.numel() for parameter in model.parameters()) - original_parameter_count,
        "trainable_parameters": trainable_parameters,
        "indic_fertility_base": base_indic_fertility,
        "indic_fertility_extended": extended_fertility,
        "indic_fertility_drop_fraction": (
            (base_indic_fertility - extended_fertility) / base_indic_fertility
        ),
        "indic_bits_per_byte_base": base_indic_bpb,
        "indic_bits_per_byte_extended": indic_bpb,
        "english_bits_per_byte_base": base_english_bpb,
        "english_bits_per_byte_extended": english_bpb,
        "english_bpb_regression": english_bpb - base_english_bpb,
        "generation_tokens_per_second": speed,
        "generation_speed_change_fraction": (speed - baseline_speed) / baseline_speed if baseline_speed else None,
        "generation_tokens_per_second_base": baseline_speed,
        "wall_hours": wall_seconds / 3600,
        "gpu_hours": wall_seconds / 3600 if args.device.startswith("cuda") else 0.0,
        "output": str(output_path),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
