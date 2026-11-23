import tokentax_lab.reporting.receipts as receipts
from tokentax_lab.reporting.receipts import (
    best_tokenizer,
    compare_tokenizers,
    prompt_receipt,
    token_receipt,
)


def test_receipt_lists_tokenizers():
    receipt = token_receipt("ನಮಸ್ಕಾರ Bangalore")
    assert len(receipt["tokenizers"]) == 2
    assert receipt["lowest_counterfactual_cost_tokenizer"] in {"cl100k_base", "o200k_base"}
    assert receipt["price"]["model"] == "gemini-4-argon"
    assert "counterfactual" in receipt["interpretation"].lower()


def test_receipt_marks_saved_tokens():
    receipt = prompt_receipt("ಕನ್ನಡ")
    for row in receipt["tokenizers"]:
        assert row["tokens_saved"] == 0


def test_receipt_reports_hf_counts_and_explicit_skips(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def load_tokenizer(model_id):
        if model_id == "google/gemma-2-2b":
            raise PermissionError("access not granted")
        return FakeTokenizer()

    monkeypatch.setattr(receipts, "_load_hf_tokenizer", load_tokenizer)
    receipt = token_receipt(
        "one two",
        tokenizer_names=("Qwen2.5", "Gemma", "unknown"),
    )
    qwen, gemma, unknown = receipt["tokenizers"]
    assert qwen["status"] == "available"
    assert qwen["tokens"] == 2
    assert gemma["status"] == "SKIPPED"
    assert "access not granted" in gemma["reason"]
    assert unknown["status"] == "SKIPPED"
    assert receipt["lowest_counterfactual_cost_tokenizer"] == "Qwen2.5"


def test_tokenizer_comparison_reports_delta_and_skipped_assets(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def load_tokenizer(model_id):
        if model_id == "google/gemma-2-2b":
            raise PermissionError("model access required")
        return FakeTokenizer()

    monkeypatch.setattr(receipts, "_load_hf_tokenizer", load_tokenizer)
    rows = compare_tokenizers(
        {"regional": "uno dos tres", "english": "one two"},
        ("cl100k_base", "Qwen2.5", "Gemma"),
    )
    assert rows[0]["regional_tokens"] is not None
    assert rows[0]["tokens_saved"] == rows[0]["regional_tokens"] - rows[0]["english_tokens"]
    assert rows[1]["status"] == "available"
    assert rows[1]["tokens_saved"] == 1
    assert rows[2]["status"] == "SKIPPED"
    assert rows[2]["regional_tokens"] is None
    assert "model access required" in rows[2]["reason"]


def test_best_tokenizer_chooses_fewest_english_tokens_and_reports_ties():
    rows = [
        {"tokenizer": "larger", "status": "available", "English translation_tokens": 8},
        {"tokenizer": "winner-a", "status": "available", "English translation_tokens": 4},
        {"tokenizer": "winner-b", "status": "available", "English translation_tokens": 4},
        {"tokenizer": "skipped", "status": "SKIPPED", "English translation_tokens": None},
    ]
    assert best_tokenizer(rows) == {
        "tokenizers": ["winner-a", "winner-b"],
        "tokens": 4,
        "tied": True,
        "text_label": "English translation",
    }


def test_best_tokenizer_returns_none_without_available_counts():
    assert best_tokenizer([
        {"tokenizer": "private", "status": "SKIPPED", "English translation_tokens": None}
    ]) is None
