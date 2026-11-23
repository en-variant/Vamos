from tokentax_lab.optimizer.safety import SafetyGate
from tokentax_lab.optimizer.features import extract_features
from tokentax_lab.optimizer.strategies import UnicodeCleanupStrategy
from tokentax_lab.optimizer.router import Router


def test_safety_gate_preserves_digits():
    gate = SafetyGate()
    ok, reason = gate.check("ಬೆಲೆ 2026", "price 2026", {})
    assert ok
    ok, reason = gate.check("ಬೆಲೆ 2026", "price 2027", {})
    assert not ok


def test_unicode_cleanup():
    out, meta = UnicodeCleanupStrategy().transform("  ಕನ್ನಡ\u200b  ")
    assert out == "ಕನ್ನಡ"


def test_features_extract_code_mix():
    features = extract_features("ನಾನು Bangaloreಗೆ bus")
    assert features["script_mix"] >= 2


def test_router_depth_is_capped():
    try:
        Router(max_depth=5)
    except ValueError as exc:
        assert "between 1 and 4" in str(exc)
    else:
        raise AssertionError("router depth above four should be rejected")


def test_round_trip_gate_rejects_changed_romanization():
    gate = SafetyGate()
    accepted, reason = gate.check(
        "ಕನ್ನಡ", "kannaDa",
        {"round_trip_kind": "transliteration", "round_trip": "ಕನ್ನಡದ"},
    )
    assert not accepted
    assert "round-trip" in reason
