from __future__ import annotations

from pathlib import Path
import json


class LanguagePackTrainer:
    """Train independent byte-level BPE packs using only supplied training text."""

    def __init__(self, vocab_size: int = 8000):
        if vocab_size < 100:
            raise ValueError("vocab_size must be at least 100")
        self.vocab_size = vocab_size

    def train(self, text_file: str | Path, output_dir: str | Path, language: str) -> dict:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        text_file = Path(text_file)
        output_dir = Path(output_dir)
        if not text_file.exists() or text_file.stat().st_size == 0:
            raise FileNotFoundError(f"Non-empty held-out training text required: {text_file}")
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / f"{language}_pack"
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train([str(text_file)], trainer)
        tokenizer.save(str(prefix) + ".json")
        return {
            "language": language,
            "tokenizer_json": str(prefix) + ".json",
            "vocab_size_actual": tokenizer.get_vocab_size(),
            "training_text": str(text_file),
        }


def merge_language_packs(base_model_id: str, pack_files: list[str], output_dir: str | Path) -> dict:
    """Add byte-level pieces from language packs to a base HF tokenizer."""
    from transformers import AutoConfig, AutoTokenizer
    from tokenizers import Tokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, use_fast=True
    )
    existing = base_tokenizer.get_vocab()
    candidates: list[str] = []
    for pack_file in pack_files:
        pack = Tokenizer.from_file(pack_file)
        candidates.extend(
            token for token in pack.get_vocab()
            if token != "[UNK]"
            and token not in existing
            and len(token) >= 2
            and any(character.isalnum() for character in token)
        )
    new_tokens = sorted(set(candidates), key=lambda token: (-len(token), token))
    added_count = base_tokenizer.add_tokens(new_tokens, special_tokens=False)
    actual_new_tokens = new_tokens[:added_count]
    base_tokenizer.save_pretrained(output_dir)
    config = AutoConfig.from_pretrained(base_model_id)
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", 0))
    tied = bool(getattr(config, "tie_word_embeddings", False))
    manifest = {
        "base_model": base_model_id,
        "output_dir": str(output_dir),
        "added_token_count": added_count,
        "new_tokens": actual_new_tokens,
        "hidden_size": hidden_size,
        "tie_word_embeddings": tied,
        "extra_parameters_after_resize": added_count * hidden_size * (1 if tied else 2),
        "pack_files": pack_files,
    }
    (output_dir / "language_pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


class MeanEmbeddingInitializer:
    """Initialize each added embedding from the base tokenizer's pieces of that text."""

    def initialize(
        self,
        model,
        base_tokenizer,
        extended_tokenizer,
        new_tokens: list[str],
        method: str = "mean",
        seed: int = 42,
    ) -> dict:
        import torch

        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        input_weight = input_embeddings.weight
        output_weight = output_embeddings.weight if output_embeddings is not None else None
        vocab = extended_tokenizer.get_vocab()
        base_size = len(base_tokenizer)
        generator = torch.Generator(device=input_weight.device).manual_seed(seed)
        initializer_range = float(getattr(model.config, "initializer_range", 0.02))
        initialized = 0
        for token in new_tokens:
            token_id = vocab[token]
            if token_id < base_size:
                continue
            if method == "random":
                value = torch.empty_like(input_weight[token_id]).normal_(
                    mean=0.0, std=initializer_range, generator=generator
                )
            elif method == "mean":
                piece_ids = base_tokenizer.encode(token, add_special_tokens=False)
                piece_ids = [piece_id for piece_id in piece_ids if 0 <= piece_id < base_size]
                if not piece_ids:
                    continue
                value = input_weight[piece_ids].detach().mean(dim=0)
            else:
                raise ValueError("method must be either 'mean' or 'random'")
            input_weight.data[token_id].copy_(value)
            if output_weight is not None and output_weight.data_ptr() != input_weight.data_ptr():
                if method == "mean" and piece_ids:
                    output_value = output_weight[piece_ids].detach().mean(dim=0)
                else:
                    output_value = value
                output_weight.data[token_id].copy_(output_value)
            initialized += 1
        return {"method": method, "initialized": initialized, "requested": len(new_tokens)}
