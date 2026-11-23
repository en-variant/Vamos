from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tokentax_lab.reporting.plots import (
    tokenizer_cost_figure,
    tokenizer_usage_figure,
)


APP_PATH = Path(__file__).parents[1] / "app.py"


def test_dashboard_renders_receipt_workbench():
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert any("Token receipt" in element.value for element in app.header)
    assert any(element.label == "Choose a workbench" for element in app.segmented_control)


def test_dashboard_switches_to_audit_without_data():
    app = AppTest.from_file(str(APP_PATH)).run()
    app.segmented_control(key="workbench").set_value("Audit").run()
    assert not app.exception
    assert any("Tokenizer observatory" in element.value for element in app.header)


def test_translation_comparison_workbench_renders_without_loading_model():
    app = AppTest.from_file(str(APP_PATH)).run()
    app.segmented_control(key="workbench").set_value("Translate & compare").run()
    assert not app.exception
    assert any("Regional language → English" in element.value for element in app.header)
    assert any(element.label == "Source language" for element in app.selectbox)
    assert any(element.label == "Regional-language text" for element in app.text_area)
    assert any(element.label == "Tokenizers to compare" for element in app.multiselect)
    assert any(element.label == "Translate and compare tokens" for element in app.button)


def test_guide_workbench_renders_detailed_usage_instructions():
    app = AppTest.from_file(str(APP_PATH)).run()
    app.segmented_control(key="workbench").set_value("Guide").run()
    assert not app.exception
    assert any("How to use TokenTax Lab" in element.value for element in app.header)


def test_translation_selects_fewest_tokenizer_and_renders_usage_chart(monkeypatch):
    from tokentax_lab.optimizer.strategies import PivotStrategy
    import tokentax_lab.reporting.receipts as receipts

    def fake_translate(self, text, language):
        return "Capital of Karnataka is Bengaluru.", {
            "added_latency_ms": 12.0,
            "quality_risk": "medium",
            "source_language": language,
            "round_trip": text,
            "round_trip_kind": "translation",
        }

    def fake_count(text, tokenizer_name):
        token_counts = {
            "cl100k_base": 10 if "ಕರ್ನಾಟಕ" in text else 8,
            "o200k_base": 7 if "ಕರ್ನಾಟಕ" in text else 9,
            "Qwen/Qwen2.5-0.5B": 6 if "ಕರ್ನಾಟಕ" in text else 5,
        }
        if tokenizer_name == "Qwen2.5":
            return token_counts["Qwen/Qwen2.5-0.5B"], "Qwen/Qwen2.5-0.5B", None
        return token_counts[tokenizer_name], None, None

    monkeypatch.setattr(PivotStrategy, "transform", fake_translate)
    monkeypatch.setattr(receipts, "_count_prompt_tokens", fake_count)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.segmented_control(key="workbench").set_value("Translate & compare").run()
    app.button(key="run_translation").click().run()

    assert not app.exception
    assert any(
        metric.label == "Fewest-token tokenizer for the English translation"
        and metric.value == "Qwen2.5"
        for metric in app.metric
    )
    assert any(
        item.label == "Select tokenizer to inspect" and item.value == "Qwen2.5"
        for item in app.selectbox
    )
    assert app.get("dataframe")


def test_token_usage_chart_plots_regional_and_english_counts():
    figure = tokenizer_usage_figure([
        {
            "tokenizer": "example",
            "status": "available",
            "regional_tokens": 12,
            "English translation_tokens": 8,
        },
        {
            "tokenizer": "private",
            "status": "SKIPPED",
            "regional_tokens": None,
            "English translation_tokens": None,
        },
    ])
    assert [trace.name for trace in figure.data] == ["Regional text", "English translation"]
    assert list(figure.data[0].y) == [12]
    assert list(figure.data[1].y) == [8]


def test_tokenizer_cost_chart_uses_configured_rate_and_excludes_skipped():
    figure = tokenizer_cost_figure(
        [
            {
                "tokenizer": "available",
                "status": "available",
                "regional_tokens": 12,
                "English translation_tokens": 8,
            },
            {
                "tokenizer": "private",
                "status": "SKIPPED",
                "regional_tokens": None,
                "English translation_tokens": None,
            },
        ],
        usd_per_million_input_tokens=2.0,
        usd_to_inr=90.0,
        price_model="configured-model",
    )
    assert [trace.name for trace in figure.data] == ["Regional text", "English translation"]
    assert list(figure.data[0].y) == [pytest.approx(0.00216)]
    assert list(figure.data[1].y) == [pytest.approx(0.00144)]
    assert "configured-model" in figure.layout.title.text
