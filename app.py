from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from tokentax_lab.audit.engine import AuditEngine, default_tokenizer_specs
from tokentax_lab.audit.flores import load_from_huggingface, load_local_flores
from tokentax_lab.data import load_config
from tokentax_lab.failure_lab import examples
from tokentax_lab.optimizer.safety import SafetyGate
from tokentax_lab.optimizer.strategies import (
    PivotStrategy,
    RomanizeStrategy,
    UnicodeCleanupStrategy,
)
from tokentax_lab.reporting.plots import (
    heatmap_figure,
    pareto_figure,
    tokenizer_cost_figure,
    tokenizer_usage_figure,
)
from tokentax_lab.reporting.receipts import (
    best_tokenizer,
    compare_tokenizers,
    prompt_receipt,
)
from tokentax_lab.reporting.benchmark import reproduce_12x

load_dotenv()
CONFIG = load_config()
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


@st.cache_resource
def local_pivot_strategy() -> PivotStrategy:
    return PivotStrategy()


st.set_page_config(page_title="TokenTax Lab", page_icon=":material/token:", layout="wide")
hero, project = st.columns([1, 8], vertical_alignment="center")
with hero:
    st.image(ROOT / "assets/grass_block.svg", width=72, alt="Pixel grass block")
with project:
    st.title("TOKEN TAX LAB")
    st.caption("A field kit for measuring language-model token tax.")

with st.sidebar:
    st.header("Field notes")
    with st.container(border=True):
        st.caption("WORKING HYPOTHESIS")
        st.write(
            "For fixed hosted tokenizers, route safe input rewrites; for open models, "
            "add language-specific vocabulary. Measure every quality and compute trade-off."
        )
    st.subheader("World status")
    st.badge(f"FLORES {'ready' if (ROOT / 'data/flores/kn.devtest').exists() else 'locked / missing'}",
             icon=":material/database:", color="green" if (ROOT / "data/flores/kn.devtest").exists() else "orange")
    st.badge(f"Belebele {'ready' if (ROOT / 'data/belebele/kn.jsonl').exists() else 'not downloaded'}",
             icon=":material/quiz:", color="green" if (ROOT / "data/belebele/kn.jsonl").exists() else "orange")
    st.caption(f"HF access: {'ready' if os.getenv('HF_TOKEN') else 'not configured'}")
    st.caption(
        f"KodeKloud OpenAI-compatible API: "
        f"{'ready' if (os.getenv('OPENAI_API_KEY') or os.getenv('OPENAPI_API_KEY')) else 'not configured'}"
    )
    if os.getenv("OPENAI_MODEL"):
        st.caption(f"API model: {os.environ['OPENAI_MODEL']}")
    st.caption(
        f"Price: {CONFIG['cost']['model']} USD "
        f"{CONFIG['cost']['usd_per_million_input_tokens']}/M input, "
        f"recorded {CONFIG['cost']['price_recorded_date']}; "
        f"FX USD/INR {CONFIG['cost']['usd_to_inr']} on {CONFIG['cost']['fx_date']}."
    )

section = st.segmented_control(
    "Choose a workbench",
    ["Receipt", "Translate & compare", "Audit", "Evaluation", "Ledger", "Failure lab", "Guide"],
    default="Receipt",
    key="workbench",
    width="stretch",
)

if section == "Receipt":
    st.header("Token receipt", icon=":material/receipt_long:")
    prompt = st.text_area("Paste a prompt", "ನನ್ನ ಬ್ಯಾಂಕ್ ಖಾತೆಯಲ್ಲಿ ₹1,25,000.50 ಇದೆ.", key="receipt_prompt")
    receipt_tokenizers = st.multiselect(
        "Receipt tokenizers",
        ["cl100k_base", "o200k_base", *default_tokenizer_specs().keys()],
        default=["cl100k_base", "o200k_base"],
        help="Select model tokenizers to try. Gated or unavailable assets are marked SKIPPED with a reason.",
    )
    if st.button("Generate receipt", type="primary", key="generate_receipt"):
        receipt = prompt_receipt(
            prompt,
            tokenizer_names=tuple(receipt_tokenizers),
            results_dir=RESULTS,
        )
        st.session_state["current_receipt"] = receipt
    receipt = st.session_state.get("current_receipt")
    if receipt:
        left, right = st.columns(2)
        with left:
            metrics = st.columns(3)
            metrics[0].metric("Strategy", receipt["strategy"])
            metrics[1].metric("Quality risk", receipt["quality_risk"])
            metrics[2].metric("Added latency", f"{receipt['added_latency_ms']:.2f} ms")
            receipt_rows = [
                {
                    "tokenizer": row["tokenizer"],
                    "status": row["status"],
                    "tokens": row["tokens"],
                    "tokens saved": row["tokens_saved"],
                    "estimated INR": row["estimated_cost_inr"],
                    "reason": row["reason"],
                }
                for row in receipt["tokenizers"]
            ]
            st.dataframe(
                pd.DataFrame(receipt_rows),
                width="stretch",
                hide_index=True,
                alt="Token count and counterfactual cost by tokenizer",
            )
            st.caption(receipt["interpretation"])
            if receipt["fallback_reason"]:
                st.warning(f"Native fallback: {receipt['fallback_reason']}")
            else:
                st.caption(receipt["router_reason"])
            if not (ROOT / "results/router_kn.joblib").exists():
                st.caption("No trained router artifact yet; receipt falls back to native.")
        with right:
            tokenizer_name = st.selectbox("Heatmap tokenizer", ["cl100k_base", "o200k_base"])
            st.plotly_chart(
                heatmap_figure(prompt, tokenizer_name),
                width="stretch",
                alt="Token-by-token segmentation for the pasted prompt",
            )
        st.caption("Gemini's token counter is the authority for actual Gemini billing.")
    else:
        st.info("Generate a receipt to see tokenizer counts, estimated rupees, savings, safety fallback, and latency.")

elif section == "Translate & compare":
    st.header("Regional language → English", icon=":material/translate:")
    st.write(
        "Translate locally with NLLB-200, then compare how selected tokenizers encode "
        "the original text and its English translation. No API key is used for this workbench."
    )
    language_names = {"kn": "Kannada", "hi": "Hindi", "ta": "Tamil"}
    language_code = st.selectbox(
        "Source language",
        options=list(language_names),
        format_func=lambda code: f"{language_names[code]} ({code})",
        key="translation_language",
    )
    examples_by_language = {
        "kn": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.",
        "hi": "भारत की राजधानी नई दिल्ली है।",
        "ta": "தமிழ்நாட்டின் தலைநகர் சென்னை.",
    }
    source_text = st.text_area(
        "Regional-language text",
        value=examples_by_language[language_code],
        height=140,
        key="translation_input",
    )
    comparison_tokenizers = st.multiselect(
        "Tokenizers to compare",
        ["cl100k_base", "o200k_base", *default_tokenizer_specs().keys()],
        default=["cl100k_base", "o200k_base", "Qwen2.5"],
        key="translation_tokenizers",
        help="Hugging Face tokenizers download on first use; gated tokenizers may be skipped.",
    )
    st.caption(
        "The first translation downloads NLLB-200 distilled (about 600M parameters) and can be slow on CPU. "
        "It runs locally after download."
    )
    if st.button("Translate and compare tokens", type="primary", key="run_translation"):
        st.session_state.pop("translation_result", None)
        if not source_text.strip():
            st.error("Enter text to translate.")
        elif not comparison_tokenizers:
            st.error("Select at least one tokenizer.")
        else:
            with st.spinner("Loading local NLLB model and translating; first use downloads model files..."):
                translated, metadata = local_pivot_strategy().transform(
                    source_text.strip(), language_code
                )
            if metadata.get("fallback"):
                st.error(f"Local translation failed: {metadata['reason']}")
            else:
                st.session_state["translation_result"] = {
                    "source": source_text.strip(),
                    "english": translated,
                    "language": language_code,
                    "metadata": metadata,
                    "tokenizers": tuple(comparison_tokenizers),
                }

    translation_result = st.session_state.get("translation_result")
    if (
        translation_result
        and translation_result["language"] == language_code
        and translation_result["source"] == source_text.strip()
    ):
        translated = translation_result["english"]
        source = translation_result["source"]
        metadata = translation_result["metadata"]
        source_column, english_column = st.columns(2)
        with source_column:
            st.text_area("Original text", value=source, height=140, disabled=True)
        with english_column:
            st.text_area("English translation", value=translated, height=140, disabled=True)

        accepted, safety_reason = SafetyGate().check(source, translated, metadata)
        if accepted:
            st.success("Round-trip, digits, and detectable entity checks passed.")
        else:
            st.warning(
                f"Translation is shown, but the safety gate did not accept it: {safety_reason}. "
                "Check names, numbers, and meaning before relying on it."
            )
        st.caption(
            f"Model: {metadata.get('model_id', 'facebook/nllb-200-distilled-600M')} · "
            f"translation + back-translation latency: {metadata['added_latency_ms']:.0f} ms"
        )
        comparison_rows = compare_tokenizers(
            {"regional": source, "English translation": translated},
            translation_result["tokenizers"],
        )
        st.subheader("Tokenizer comparison")
        recommendation = best_tokenizer(comparison_rows)
        available_rows = [row for row in comparison_rows if row["status"] == "available"]
        if recommendation:
            winner_label = ", ".join(recommendation["tokenizers"])
            st.metric(
                "Fewest-token tokenizer for the English translation",
                winner_label,
                f"{recommendation['tokens']} tokens"
                + (" (tie)" if recommendation["tied"] else ""),
            )
            st.caption(
                "This is a token-count recommendation only. A hosted model's tokenizer is fixed; "
                "selecting a different tokenizer here does not change that model or its billing."
            )
            selected_name = st.selectbox(
                "Select tokenizer to inspect",
                options=[row["tokenizer"] for row in available_rows],
                index=next(
                    index for index, row in enumerate(available_rows)
                    if row["tokenizer"] in recommendation["tokenizers"]
                ),
                key="translation_selected_tokenizer",
            )
            selected_row = next(
                row for row in available_rows if row["tokenizer"] == selected_name
            )
            st.caption(
                f"Selected: {selected_name} · English translation uses "
                f"{selected_row['English translation_tokens']} tokens."
            )
            st.plotly_chart(
                tokenizer_usage_figure(comparison_rows),
                width="stretch",
                alt="Regional and translated English token counts by tokenizer",
            )
            price = CONFIG["cost"]
            api_input_rate = os.getenv("OPENAI_USD_INPUT_PER_MILLION")
            api_usd_to_inr = os.getenv("OPENAI_USD_TO_INR")
            if api_input_rate and api_usd_to_inr:
                cost_model = os.getenv("OPENAI_MODEL") or price["model"]
                cost_rate = float(api_input_rate)
                cost_fx = float(api_usd_to_inr)
                cost_basis = (
                    f"OpenAI-compatible API input rate for {cost_model} from local "
                    ".env; verify against the provider's current rate card."
                )
            else:
                cost_model = str(price["model"])
                cost_rate = float(price["usd_per_million_input_tokens"])
                cost_fx = float(price["usd_to_inr"])
                cost_basis = (
                    f"fallback reference rate for {cost_model}: "
                    f"price recorded {price['price_recorded_date']}, "
                    f"USD/INR {cost_fx} from {price['fx_date']}."
                )
            st.plotly_chart(
                tokenizer_cost_figure(
                    comparison_rows,
                    cost_rate,
                    cost_fx,
                    cost_model,
                ),
                width="stretch",
                alt="Estimated regional and English input cost in rupees by tokenizer",
            )
            st.caption(
                f"Counterfactual input-only estimate using USD {cost_rate}/million tokens; "
                f"{cost_basis} "
                "The same rate is applied to every tokenizer; these are not actual provider "
                "token counts, bills, or translation-quality measurements."
            )
        else:
            st.warning("No selected tokenizer was available to count this text.")
        st.dataframe(
            pd.DataFrame(comparison_rows),
            width="stretch",
            hide_index=True,
            alt="Token counts for original regional text and English translation",
        )
        st.caption(
            "Positive “tokens saved” means the English version uses fewer tokens; "
            "negative means it uses more. Counts compare these exact texts, not translation quality."
        )
        skipped_rows = [row for row in comparison_rows if row["status"] != "available"]
        if skipped_rows:
            with st.expander("Why some tokenizers were skipped"):
                st.dataframe(
                    pd.DataFrame(skipped_rows)[["tokenizer", "model_id", "reason"]],
                    width="stretch",
                    hide_index=True,
                    alt="Reasons model tokenizers were unavailable",
                )

elif section == "Audit":
    st.header("Tokenizer observatory", icon=":material/troubleshoot:")
    st.write("The gated FLORES devtest must be accessed under its evaluation-only terms. It is never used for vocabulary training.")
    if st.button("Download FLORES-200 devtest"):
        try:
            with st.spinner("Fetching the gated 1,012-sentence parallel devtest..."):
                loaded = load_from_huggingface(ROOT / "data/flores")
            st.success(f"Downloaded {len(loaded)} languages × 1,012 sentences.")
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"FLORES download failed: {type(exc).__name__}: {exc}")
    if st.button("Run tokenizer audit"):
        try:
            language_texts = load_local_flores(ROOT / "data/flores", ("kn", "hi", "ta", "en"))
            price = CONFIG["cost"]
            engine = AuditEngine(
                context_tokens=CONFIG["context_tokens"],
                price_model=price["model"],
                price_usd_per_million=price["usd_per_million_input_tokens"],
                usd_to_inr=price["usd_to_inr"],
                price_recorded_date=price["price_recorded_date"],
                fx_date=price["fx_date"],
            )
            rows = engine.audit_many(("cl100k_base", "o200k_base"), language_texts)
            ratios = [
                reproduce_12x(language_texts, encoding)
                for encoding in ("cl100k_base", "o200k_base")
            ]
            for name, model_id in default_tokenizer_specs().items():
                rows.extend(engine.audit_hf(model_id, language_texts, name))
            RESULTS.mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(RESULTS / "audit.csv", index=False)
            (RESULTS / "audit.json").write_text(
                json.dumps({
                    "rows": rows,
                    "price": price,
                    "12x_claim_reproduction": ratios,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success("Saved measured audit to results/audit.csv and results/audit.json")
            st.dataframe(
                pd.DataFrame(ratios),
                width="stretch",
                alt="Measured Kannada to English token ratios",
            )
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"Audit failed: {type(exc).__name__}: {exc}")
    audit_path = RESULTS / "audit.csv"
    if audit_path.exists():
        st.dataframe(pd.read_csv(audit_path), width="stretch", alt="Tokenizer audit metrics by language")
    else:
        st.info("No audit results yet. The cost/quality figures below will not be fabricated.")

elif section == "Evaluation":
    st.header("Trial board", icon=":material/science:")
    selected_language = st.selectbox("Language", ["kn", "hi", "ta", "en"], key="frontier_language")
    evaluation_path = RESULTS / f"evaluation_{selected_language}.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        points = evaluation.get("summary", {}).get("points", [])
        if points:
            st.plotly_chart(
                pareto_figure(points),
                width="stretch",
                alt="Measured strategy accuracy versus average prompt tokens",
            )
            st.dataframe(pd.DataFrame(points), width="stretch", alt="Measured strategy results and confidence intervals")
            st.caption("Only measured points are shown. Confidence intervals use passage-cluster bootstrap.")
            if evaluation.get("router"):
                st.json(evaluation["router"])
        else:
            st.warning("The saved run contains no complete comparable strategy results.")
        st.download_button(
            "Download raw evaluation report",
            evaluation_path.read_bytes(),
            file_name=evaluation_path.name,
            mime="application/json",
        )
    else:
        st.info("Download Belebele and run `tokentax evaluate`. The chart stays empty until actual measurements exist.")

elif section == "Ledger":
    st.header("Upgrade ledger", icon=":material/construction:")
    adaptation_files = list(RESULTS.glob("adapted-*/metrics.json"))
    if adaptation_files:
        ledger_rows = []
        for path in adaptation_files:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            ledger_rows.append({
                "improvement": path.parent.name,
                "fertility drop": metrics["indic_fertility_drop_fraction"],
                "Indic BPB base → extended": (
                    f"{metrics['indic_bits_per_byte_base']:.4f} → "
                    f"{metrics['indic_bits_per_byte_extended']:.4f}"
                ),
                "English BPB regression": metrics["english_bpb_regression"],
                "extra parameters": metrics["extra_parameters"],
                "wall hours": metrics["wall_hours"],
                "GPU-hours": metrics["gpu_hours"],
                "generation tokens/s": metrics["generation_tokens_per_second"],
            })
        st.dataframe(pd.DataFrame(ledger_rows), width="stretch", alt="Tokenizer improvements and measured costs")
    else:
        st.info("No tokenizer adaptation run has produced metrics yet.")
    st.caption("Tokenizer adaptation and generation-speed measurements are emitted by scripts/train_embedding_rows.py.")

elif section == "Failure lab":
    st.header("Failure lab", icon=":material/bug_report:")
    for example in examples():
        with st.expander(example["name"]):
            st.code(example["native"])
            st.write(example["breaking"])
            strategy = (
                UnicodeCleanupStrategy()
                if example["name"] == "Unicode zero width"
                else RomanizeStrategy("itrans")
            )
            rewritten, metadata = strategy.transform(example["native"], "kn")
            accepted, reason = SafetyGate().check(example["native"], rewritten, metadata)
            st.write(f"Rewrite: {'accepted' if accepted else 'native fallback'} — {reason}")
            st.code(rewritten if accepted else example["native"])

elif section == "Guide":
    st.header("How to use TokenTax Lab", icon=":material/menu_book:")
    st.write(
        "Start with a single prompt, then run dataset-backed workflows for evidence. "
        "The app never fills missing benchmark results with sample numbers."
    )
    with st.expander("1. Translate a regional-language sentence and compare tokenizers", expanded=True):
        st.markdown(
            """
            1. Open **Translate & compare**.
            2. Choose Kannada, Hindi, or Tamil and enter a sentence.
            3. Choose tokenizers. `cl100k_base` and `o200k_base` are immediately available;
               Hugging Face tokenizers may download on first use and gated assets may be skipped.
            4. Click **Translate and compare tokens**. NLLB-200 runs locally, including a
               back-translation for the safety check; the first run downloads model files.
            5. Read the English output and safety notice. The grouped charts compare original
               and English token counts and estimated input cost; the recommendation names the
               tokenizer(s) with the fewest English tokens. Use the selector to inspect another.

            This recommendation only compares token counts for these exact strings. A hosted
            provider's tokenizer is fixed to its model; the selector does not change API billing
            or model behavior. Rupee costs are counterfactual estimates at the configured input
            rate, not actual provider charges or translation-quality measurements.
            """
        )
    with st.expander("2. Create a token receipt and inspect a split"):
        st.markdown(
            """
            1. Open **Receipt**, paste the full prompt you plan to send, and select tokenizers.
            2. Click **Generate receipt** to see counts, token savings, estimated costs, and latency.
            3. Use **Heatmap tokenizer** to inspect token boundaries.

            Receipt prices for non-provider tokenizers are counterfactual estimates, not actual API
            charges. The receipt only uses a trained router when its artifact exists; otherwise it
            selects native text.
            """
        )
    with st.expander("3. Run the tokenizer audit"):
        st.markdown(
            """
            1. Accept FLORES-200 dataset access terms and set `HF_TOKEN` in the ignored local `.env`.
            2. Open **Audit** and click **Download FLORES-200 devtest**.
            3. Click **Run tokenizer audit**. It compares 1,012 parallel devtest sentences per
               language and writes measured CSV/JSON artifacts under `results/`.

            FLORES is evaluation-only. Do not use it to train vocabulary packs or redistribute it.
            """
        )
    with st.expander("4. Run quality evaluation and inspect the frontier"):
        st.markdown(
            """
            Download public Belebele with `tokentax download --belebele`. Then configure the API
            endpoint/key/model/rates in `.env` and run:

            ```bash
            tokentax evaluate --backend openai --allow-paid --budget-cap-inr 50 --language kn
            ```

            Repeat for `hi`, `ta`, and `en` as needed. This makes paid API calls. The budget guard's
            preflight token count is an estimate; check provider usage/billing separately. Router
            and Pareto results appear in **Evaluation** only after successful evaluation runs.
            """
        )
    with st.expander("5. Read the ledger and failure lab"):
        st.markdown(
            """
            - **Ledger** shows measured adaptation costs only after tokenizer training/adaptation
              has produced metrics.
            - **Failure lab** applies rewrite strategies to examples and reports safety fallbacks.
            - `not measured`, `SKIPPED`, and `blocked` are real statuses; they are not zero results.
            """
        )
