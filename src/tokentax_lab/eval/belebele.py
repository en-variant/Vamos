from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Protocol

import tiktoken

from ..optimizer.safety import SafetyGate
from ..optimizer.strategies import NativeStrategy, Strategy


class Generator(Protocol):
    def generate(self, prompt: str) -> tuple[str, dict]: ...


class GeminiGenerator:
    """Gemini API adapter with preflight budget reservation and actual usage accounting."""

    def __init__(
        self,
        model_id: str = "gemini-4-argon",
        budget_cap_inr: float = 50.0,
        usd_input_per_million: float = 2.0,
        usd_output_per_million: float = 10.0,
        usd_to_inr: float = 92.68,
        max_output_tokens: int = 16,
    ):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY locally before using Gemini API evaluation.")
        if budget_cap_inr < 0:
            raise ValueError("budget_cap_inr cannot be negative")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.budget_cap_inr = budget_cap_inr
        self.usd_input_per_million = usd_input_per_million
        self.usd_output_per_million = usd_output_per_million
        self.usd_to_inr = usd_to_inr
        self.max_output_tokens = max_output_tokens
        self.spent_inr = 0.0
        self.reserved_inr = 0.0
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        result = self.client.models.count_tokens(model=self.model_id, contents=text)
        return int(result.total_tokens)

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        usd = (
            input_tokens * self.usd_input_per_million
            + output_tokens * self.usd_output_per_million
        ) / 1_000_000
        return usd * self.usd_to_inr

    def generate(self, prompt: str) -> tuple[str, dict]:
        from google.genai import types

        input_tokens = self.count_tokens(prompt)
        request_reserve = self._cost(input_tokens, self.max_output_tokens)
        if self.spent_inr + self.reserved_inr + request_reserve > self.budget_cap_inr:
            raise RuntimeError(
                f"Budget limit reached before request: spent ₹{self.spent_inr:.4f}, "
                f"request reserve ₹{request_reserve:.4f}, cap ₹{self.budget_cap_inr:.2f}"
            )
        self.reserved_inr += request_reserve
        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception:
            self.reserved_inr -= request_reserve
            raise
        self.reserved_inr -= request_reserve
        usage = getattr(response, "usage_metadata", None)
        actual_input = int(getattr(usage, "prompt_token_count", input_tokens) or input_tokens)
        candidate_tokens = getattr(usage, "candidates_token_count", None)
        thought_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        actual_output = (
            int(candidate_tokens) + int(thought_tokens)
            if candidate_tokens is not None
            else self.max_output_tokens
        )
        actual_cost = self._cost(actual_input, actual_output)
        self.spent_inr += actual_cost
        self.calls += 1
        return response.text or "", {
            "prompt_tokens": actual_input,
            "completion_tokens": actual_output,
            "cost_inr": actual_cost,
            "spent_inr": self.spent_inr,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "model": self.model_id,
        }


class OpenAICompatibleGenerator:
    """OpenAI-compatible chat-completions client with a local estimated budget guard."""

    def __init__(
        self,
        model_id: str | None = None,
        budget_cap_inr: float = 50.0,
        max_output_tokens: int = 16,
    ):
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAPI_API_KEY")
        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAPI_BASE_URL")
            or "https://api.ai.kodekloud.com/v1"
        )
        model_id = model_id or os.environ.get("OPENAI_MODEL")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in the local .env file before API evaluation.")
        if not model_id:
            raise RuntimeError("Set OPENAI_MODEL to a model enabled for your KodeKloud API key.")
        try:
            input_rate = float(os.environ["OPENAI_USD_INPUT_PER_MILLION"])
            output_rate = float(os.environ["OPENAI_USD_OUTPUT_PER_MILLION"])
            usd_to_inr = float(os.environ["OPENAI_USD_TO_INR"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "Set OPENAI_USD_INPUT_PER_MILLION, OPENAI_USD_OUTPUT_PER_MILLION, "
                "and OPENAI_USD_TO_INR from the provider's current rate card."
            ) from exc
        if min(input_rate, output_rate) < 0 or usd_to_inr <= 0:
            raise ValueError("API token rates must be non-negative and USD_TO_INR must be positive")
        if budget_cap_inr < 0:
            raise ValueError("budget_cap_inr cannot be negative")

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.budget_cap_inr = budget_cap_inr
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.usd_to_inr = usd_to_inr
        self.max_output_tokens = max_output_tokens
        self.encoding = tiktoken.get_encoding(os.environ.get("OPENAI_TOKENIZER", "o200k_base"))
        self.spent_inr = 0.0
        self.reserved_inr = 0.0
        self.calls = 0

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        usd = (
            input_tokens * self.input_rate + output_tokens * self.output_rate
        ) / 1_000_000
        return usd * self.usd_to_inr

    def generate(self, prompt: str) -> tuple[str, dict]:
        estimated_input_tokens = len(self.encoding.encode(prompt))
        estimated_reserve = self._cost(estimated_input_tokens, self.max_output_tokens)
        if self.spent_inr + self.reserved_inr + estimated_reserve > self.budget_cap_inr:
            raise RuntimeError(
                f"Budget limit reached before request: spent ₹{self.spent_inr:.4f}, "
                f"estimated request reserve ₹{estimated_reserve:.4f}, "
                f"cap ₹{self.budget_cap_inr:.2f}"
            )
        self.reserved_inr += estimated_reserve
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=self.max_output_tokens,
            )
        except Exception:
            self.reserved_inr -= estimated_reserve
            raise
        self.reserved_inr -= estimated_reserve

        usage = getattr(response, "usage", None)
        actual_input = int(getattr(usage, "prompt_tokens", estimated_input_tokens) or estimated_input_tokens)
        actual_output = int(getattr(usage, "completion_tokens", self.max_output_tokens) or 0)
        actual_cost = self._cost(actual_input, actual_output)
        self.spent_inr += actual_cost
        self.calls += 1
        choice = response.choices[0] if response.choices else None
        content = getattr(getattr(choice, "message", None), "content", None) if choice else None
        return content or "", {
            "prompt_tokens": actual_input,
            "completion_tokens": actual_output,
            "estimated_prompt_tokens": estimated_input_tokens,
            "cost_inr": actual_cost,
            "spent_inr": self.spent_inr,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "model": self.model_id,
            "cost_basis": (
                "Provider token usage priced at configured rate-card values; "
                "preflight cap uses local tokenizer estimate."
            ),
        }


class LocalCausalGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model_id = model_id
        self.device = device
        self.model.eval()

    def generate(self, prompt: str) -> tuple[str, dict]:
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.perf_counter()
        with self.torch.no_grad():
            output = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output[0, encoded["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text, {
            "prompt_tokens": int(encoded["input_ids"].shape[1]),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "model": self.model_id,
        }


class BelebeleEvaluator:
    """Multiple-choice scorer. Strategy changes only the passage, not the prompt template."""

    TEMPLATE = (
        "Read the passage and answer the question. Reply with exactly one option letter: A, B, C, or D.\n\n"
        "Passage:\n{passage}\n\nQuestion:\n{question}\n\n"
        "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\nAnswer:"
    )
    ANSWER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

    def __init__(self, generator: Generator, safety_gate: SafetyGate | None = None):
        self.generator = generator
        self.safety_gate = safety_gate or SafetyGate()

    def format_prompt(self, item: dict, passage: str | None = None) -> str:
        return self.TEMPLATE.format(
            passage=passage if passage is not None else item["flores_passage"],
            question=item["question"],
            a=item["mc_answer1"],
            b=item["mc_answer2"],
            c=item["mc_answer3"],
            d=item["mc_answer4"],
        )

    def control_prompt_hash(self, item: dict) -> str:
        template_only = self.format_prompt(item, passage="<STRATEGY_PASSAGE>")
        return hashlib.sha256(template_only.encode("utf-8")).hexdigest()

    def predict(
        self,
        item: dict,
        strategy: Strategy | None = None,
        language: str = "kn",
        option_order: list[int] | None = None,
    ) -> dict:
        strategy = strategy or NativeStrategy()
        source_passage = item["flores_passage"]
        passage, metadata = strategy.transform(source_passage, language)
        accepted, reason = self.safety_gate.check(source_passage, passage, metadata)
        fallback = not accepted
        if fallback:
            passage = source_passage
        prompt = self.format_prompt(item, passage)
        if option_order is not None:
            options = [item[f"mc_answer{i}"] for i in option_order]
            prompt = self.TEMPLATE.format(
                passage=passage,
                question=item["question"],
                a=options[0],
                b=options[1],
                c=options[2],
                d=options[3],
            )
        raw, generation = self.generator.generate(prompt)
        postprocess = getattr(strategy, "postprocess_response", None)
        response = postprocess(raw, metadata) if postprocess else raw
        match = self.ANSWER_RE.search(response.strip())
        predicted = match.group(1).upper() if match else None
        gold = int(item["correct_answer_num"])
        if option_order is not None:
            # option_order holds 1-based original option numbers in presented order.
            gold = option_order.index(gold) + 1
        return {
            "passage_id": item.get("link", str(item.get("question_number", ""))),
            "question_number": item.get("question_number"),
            "language": language,
            "strategy": strategy.name,
            "scheme": metadata.get("scheme"),
            "added_latency_ms": metadata.get("added_latency_ms", 0.0),
            "quality_risk": metadata.get("quality_risk", "unknown"),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "control_prompt_sha256": self.control_prompt_hash(item),
            "accepted": accepted,
            "fallback_reason": None if accepted else reason,
            "rewritten_passage": passage,
            "prediction": predicted,
            "gold": "ABCD"[gold - 1] if gold in range(1, 5) else None,
            "correct": predicted == ("ABCD"[gold - 1] if gold in range(1, 5) else None),
            "raw_output": raw,
            "response": response,
            "rewrite_latency_ms": metadata.get("added_latency_ms", 0.0),
            "end_to_end_latency_ms": (
                generation.get("latency_ms", 0.0)
                + metadata.get("added_latency_ms", 0.0)
            ),
            **generation,
        }
