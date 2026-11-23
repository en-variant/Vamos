from tokentax_lab.audit.engine import AuditEngine


def test_audit_sentence_counts():
    result = AuditEngine().audit_tiktoken("cl100k_base", "kn", ["ನಮಸ್ಕಾರ", "ಬೆಂಗಳೂರು"])
    assert result.sentences == 2
    assert result.total_tokens > 0
    assert result.tokens_per_sentence > 0


def test_ratio_generated_for_languages():
    rows = AuditEngine().audit_many(["cl100k_base"], {"kn": ["ನಮಸ್ಕಾರ"], "en": ["Hello"]})
    kn = next(r for r in rows if r["language"] == "kn")
    assert "ratio_to_english" in kn


def test_pricing_is_explicit_and_dated():
    result = AuditEngine(
        price_model="gemini-4-argon",
        price_usd_per_million=2.0,
        usd_to_inr=92.68,
        price_recorded_date="2026-11-21",
        fx_date="2026-11-20",
    ).audit_tiktoken("cl100k_base", "kn", ["ನಮಸ್ಕಾರ"])
    assert result.estimated_cost_inr is not None
    assert result.price_recorded_date == "2026-11-21"
    assert "counterfactual" in result.cost_basis.lower()
