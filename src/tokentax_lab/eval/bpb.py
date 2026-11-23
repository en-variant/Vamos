from __future__ import annotations

import math


def bits_per_byte(negative_log_likelihood: float, utf8_bytes: int) -> float:
    if utf8_bytes <= 0:
        raise ValueError("utf8_bytes must be positive")
    if negative_log_likelihood < 0:
        raise ValueError("negative_log_likelihood must be nonnegative")
    return negative_log_likelihood / (utf8_bytes * math.log(2))


def corpus_bpb(model, tokenizer, texts: list[str], device: str = "cpu") -> float:
    import torch
    import torch.nn.functional as functional

    total_nll = 0.0
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    if not texts or total_bytes == 0:
        raise ValueError("At least one non-empty text is required for BPB")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt").to(device)
            output = model(**encoded)
            logits = output.logits[:, :-1, :].contiguous()
            labels = encoded["input_ids"][:, 1:].contiguous()
            total_nll += float(functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                reduction="sum",
            ).cpu())
    return bits_per_byte(total_nll, total_bytes)
