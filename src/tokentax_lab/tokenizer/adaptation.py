from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptationCost:
    wall_hours: float
    gpu_hours: float
    extra_parameters: int
    trainable_parameters: int
    device: str


def freeze_backbone(model, new_token_ids: list[int], train_lora: bool = False) -> int:
    """Train only newly added input/output rows; mask gradients for all base rows."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(train_lora and "lora_" in name)
    trainable_parameters = 0
    seen_parameters: set[int] = set()
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    for module in (input_embeddings, output_embeddings):
        if module is None or id(module.weight) in seen_parameters:
            continue
        seen_parameters.add(id(module.weight))
        module.weight.requires_grad_(True)
        row_mask = module.weight.new_zeros((module.weight.shape[0], 1))
        row_mask[new_token_ids] = 1
        module.weight.register_hook(lambda grad, mask=row_mask: grad * mask)
        trainable_parameters += len(new_token_ids) * module.weight.shape[1]
    if train_lora:
        trainable_parameters += sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        )
    return trainable_parameters


def add_lora(model):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install the optional 'peft' package before requesting LoRA adaptation") from exc
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)
