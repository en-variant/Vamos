from types import SimpleNamespace

import pytest

from tokentax_lab.eval.belebele import (
    BelebeleEvaluator,
    GeminiGenerator,
    OpenAICompatibleGenerator,
)
from tokentax_lab.eval.benchmark import train_router_and_compare
from tokentax_lab.optimizer.strategies import NativeStrategy, UnicodeCleanupStrategy


class FakeGenerator:
    def generate(self, prompt):
        assert "Answer:" in prompt
        return "A", {"prompt_tokens": len(prompt.split()), "latency_ms": 1.0, "model": "fake"}


def sample_item(link="p1", number=1):
    return {
        "link": link,
        "question_number": number,
        "flores_passage": "Passage with 2026 in it.",
        "question": "Which is correct?",
        "mc_answer1": "one",
        "mc_answer2": "two",
        "mc_answer3": "three",
        "mc_answer4": "four",
        "correct_answer_num": "1",
    }


def test_evaluator_keeps_control_prompt_template_identical():
    evaluator = BelebeleEvaluator(FakeGenerator())
    native = evaluator.predict(sample_item(), NativeStrategy(), "en")
    cleaned = evaluator.predict(sample_item(), UnicodeCleanupStrategy(), "en")
    assert native["correct"] and cleaned["correct"]
    assert native["control_prompt_sha256"] == cleaned["control_prompt_sha256"]
    assert native["prompt_sha256"] == cleaned["prompt_sha256"]


def test_evaluator_falls_back_when_number_changed():
    class RewritingStrategy:
        name = "bad_numbers"

        def transform(self, text, language):
            return text.replace("2026", "2027"), {"quality_risk": "high"}

    result = BelebeleEvaluator(FakeGenerator()).predict(
        sample_item(), RewritingStrategy(), "en"
    )
    assert not result["accepted"]
    assert result["fallback_reason"] == "digit sequence changed"
    assert result["correct"]


def test_router_evaluation_splits_by_passage():
    items, rows = [], []
    strategies = ["native", "unicode_cleanup"]
    for passage_index in range(12):
        link = f"passage-{passage_index}"
        for question in (1, 2):
            item = sample_item(link, question)
            items.append(item)
            for strategy in strategies:
                rows.append({
                    "item_id": f"kn|{link}|{question}",
                    "passage_id": link,
                    "strategy": strategy,
                    "scheme": None,
                    "correct": strategy == "native" or passage_index % 2 == 0,
                    "prompt_tokens": 40 if strategy == "native" else 30,
                    "cost_inr": 0.001,
                })
    result = train_router_and_compare(
        {"results": rows, "strategies": strategies},
        items,
        "kn",
        quality_floor=0.25,
    )
    assert result["router_depth"] <= 4
    assert set(result["train_passages"]).isdisjoint(result["test_passages"])
    assert result["test_comparison"]["router"]["n"] > 0


def test_gemini_budget_preflight_stops_before_api_call(monkeypatch):
    import google.genai

    class Models:
        called = False

        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=100)

        def generate_content(self, **kwargs):
            self.called = True
            raise AssertionError("request must be blocked by zero budget")

    models = Models()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-a-secret")
    monkeypatch.setattr(
        google.genai, "Client",
        lambda **kwargs: SimpleNamespace(models=models),
    )
    generator = GeminiGenerator(budget_cap_inr=0.0)
    with pytest.raises(RuntimeError, match="Budget limit reached"):
        generator.generate("A small test prompt")
    assert not models.called


def test_openai_compatible_generator_requires_key_model_and_rates(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAPI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatibleGenerator()

    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        OpenAICompatibleGenerator()


def test_openai_compatible_generator_budget_and_usage_accounting(monkeypatch):
    import sys

    requests = []
    client_options = {}

    class ChatCompletions:
        called = False

        def create(self, **kwargs):
            self.called = True
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="A"))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=1),
            )

    completions = ChatCompletions()
    def create_client(**kwargs):
        client_options.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    fake_openai = SimpleNamespace(
        OpenAI=create_client
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAPI_API_KEY", "local-test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAPI_BASE_URL", "https://api.ai.kodekloud.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_USD_INPUT_PER_MILLION", "1")
    monkeypatch.setenv("OPENAI_USD_OUTPUT_PER_MILLION", "2")
    monkeypatch.setenv("OPENAI_USD_TO_INR", "90")

    blocked = OpenAICompatibleGenerator(budget_cap_inr=0.0)
    with pytest.raises(RuntimeError, match="Budget limit reached"):
        blocked.generate("A small test prompt")
    assert not completions.called

    generator = OpenAICompatibleGenerator(budget_cap_inr=1.0)
    text, metadata = generator.generate("A small test prompt")
    assert text == "A"
    assert requests[0]["model"] == "test-model"
    assert client_options["api_key"] == "local-test-key"
    assert client_options["base_url"] == "https://api.ai.kodekloud.com/v1"
    assert metadata["prompt_tokens"] == 20
    assert metadata["completion_tokens"] == 1
    assert metadata["cost_inr"] == pytest.approx((20 + 2) / 1_000_000 * 90)
