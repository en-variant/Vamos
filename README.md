# TokenTax Lab

TokenTax Lab measures the tokenization cost gap for Kannada (`kn`), Hindi
(`hi`), and Tamil (`ta`) against English (`en`). It keeps measured results
separate from estimates and blocked experiments.

## Hypothesis

For hosted models with fixed tokenizers, input-side rewriting chosen per input
by a lightweight router lowers billed tokens at a bounded quality loss; for
open models, per-language vocabulary extension removes most of the gap at its
source. Both approaches are evaluated on a shared cost-quality frontier.

## Quick start

Python 3.10+ is supported for the core app. The optional PyTorch model workflow
may require a Python version with a compatible PyTorch wheel and enough local
memory/storage for the selected model.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
streamlit run app.py
```

The `.env` file is optional for the receipt, public Belebele download, and
core UI. Never paste credentials into source code or commit `.env` or
`.streamlit/secrets.toml`.

## Using the Streamlit app

Launch from the project directory:

```bash
cd /home/en_variant/Vamos
streamlit run app.py
```

Choose a workbench from the top selector:

- **Receipt**: paste the exact prompt, choose tokenizers, click **Generate
  receipt**, and inspect count, counterfactual cost, router choice, latency,
  safety fallback, and the token heatmap. No trained router means native
  input is used.
- **Translate & compare**: choose Kannada, Hindi, or Tamil, enter text, select
  tokenizers, then click **Translate and compare tokens**. NLLB-200 distilled
  runs locally and downloads its model on first use; expect a large download
  and slower CPU inference. It also back-translates to check round-trip
  similarity and protected numbers/entities. The translation is shown even if
  the gate warns it should not be trusted without review. Grouped bar charts
  compare original and English token counts and their counterfactual input
  costs for each available tokenizer. Cost uses the configured input-token
  rate and USD/INR conversion, and is not an actual provider bill.
  The recommended tokenizer is the one with the fewest pieces for the English
  text; ties are reported, and you can select an available tokenizer to inspect.
  Positive saved-token counts mean English used fewer tokens.
- **Audit**: download gated FLORES after accepting its terms and configuring
  `HF_TOKEN`; then run the audit. It writes results locally under `results/`.
- **Evaluation**: run Belebele evaluation from the CLI (see below); this
  workbench displays saved measured frontier points, not predictions.
- **Ledger**: shows tokenizer adaptation metrics only after those experiments
  have generated metric files.
- **Failure lab**: inspect examples where rewrites may be unsafe.
- **Guide**: in-app instructions for the main workflows.

Tokenizer entries marked `SKIPPED` could not be loaded (for example because a
model is gated or the machine is offline). Receipt/token-count estimates do
not measure semantic translation quality or actual provider billing.

### Input-side translation in the UI

The **Translate & compare** workbench is a local interactive translation
utility, not a benchmark. It uses `facebook/nllb-200-distilled-600M`; it does
not require `HF_TOKEN` for a public model, though first use needs network access
to download model files. On CPU, a short sentence can take noticeable time.
The benchmark workflow below separately evaluates strategies on Belebele.
The tokenizer recommendation is only a token-count comparison. A hosted
service's tokenizer is fixed to its model, so choosing a different tokenizer in
this UI does not change that service's tokenization, quality, or bill.

## Reproducible workflow

### 1. Download public evaluation data

Belebele is public and pinned to a dataset revision. This command downloads
four 900-question language files and validates their schema:

```bash
tokentax download --belebele
```

### 2. Run the tokenizer audit

FLORES-200 devtest is gated. First accept its Hugging Face terms, then set
`HF_TOKEN` in the local environment or `.env`; alternatively, provide licensed
local files. The loader requires exactly 1,012 parallel devtest sentences per
language. FLORES is evaluation-only and is never used to train tokenizer
packs.

```bash
tokentax download --flores
tokentax audit --flores data/flores --output results/audit
```

The audit includes `cl100k_base`, `o200k_base`, and tokenizer assets that can
be loaded locally. Inaccessible model tokenizers are reported as skipped.
Rupee estimates for non-Gemini tokenizers are explicitly counterfactual, using
the recorded Gemini rate and USD/INR conversion; they are not actual Gemini
billing counts. The receipt's counts are not a substitute for Gemini's
`countTokens` endpoint.

### 3. Evaluate rewrites and the router

For CPU-only Linux, install PyTorch from its CPU wheel index first so pip does
not pull the much larger CUDA build, then install the project model extra:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 'torch>=2.2'
python -m pip install -e '.[model]'
tokentax evaluate --backend local --model Qwen/Qwen2.5-0.5B --language kn
```

For GPU use, install the PyTorch wheel matching the machine's CUDA setup from
the official PyTorch selector before installing `.[model]`.

The full evaluation is intentionally substantial. `--limit N` can be used
for a clearly labeled pilot; it is not a full benchmark. English calibration
must pass its 95% passage-bootstrap confidence-interval floor before results
are reported. The report compares native, cleanup, mixed-language, three
romanization schemes, pivot translation, and a passage-held-out router.
Router splits are by passage, and quality intervals are cluster-bootstrapped.
Local pivot translation requires the model download and can be slow on CPU.

The OpenAI-compatible KodeKloud endpoint can be configured locally with
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. Get the model name and
current per-million input/output rates from the provider's API/model page; the
client refuses to call the endpoint until the model, rate-card values, and
USD/INR conversion are configured. Its ₹50 preflight budget guard uses a local
tokenizer estimate (selectable with `OPENAI_TOKENIZER`); actual reported cost
uses token usage from the API response and the configured rate card. The
estimate is not a provider-guaranteed token count.

Put credentials in `.env`, never `.env.example`. The backend accepts
`OPENAPI_API_KEY` and `OPENAPI_BASE_URL` as aliases, but `OPENAI_API_KEY` and
`OPENAI_BASE_URL` are the preferred variable names.

```bash
tokentax evaluate --backend openai --allow-paid --budget-cap-inr 50 --language kn
```

Gemini evaluation remains an optional paid backend. Set `GOOGLE_API_KEY`
locally and explicitly select the Gemini backend:

```bash
tokentax evaluate --backend gemini --allow-paid --budget-cap-inr 50 --language kn
```

The evaluator preflights request cost and stops at the configured cap. Do not
raise the cap without an explicit decision. No paid calls are required for
the basic app or data-download checks.

### 4. Train and measure vocabulary packs

This workflow uses distinct Wikimedia document slices, not FLORES:

```bash
python scripts/download_heldout_text.py --train-docs 5000 --eval-docs 500
python scripts/train_language_packs.py --vocab-size 8000
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 'torch>=2.2'
python -m pip install -e '.[model]'
python scripts/train_embedding_rows.py --init mean
python scripts/train_embedding_rows.py --init random --output results/adapted-random
```

The adaptation script reports fertility, Indic and English bits-per-byte,
parameter counts, wall/GPU hours, and generation speed. BPB is used for
cross-tokenizer quality comparisons; per-token perplexity is not comparable.
This training path downloads a base model and is compute intensive. The
optional `--lora` flag also requires the optional PEFT dependency; install
the `lora` extra before using it:

```bash
python -m pip install -e '.[lora]'
```

### 5. Check readiness and launch the app

```bash
tokentax status
tokentax receipt --text "ನನ್ನ ಬ್ಯಾಂಕ್ ಖಾತೆಯಲ್ಲಿ ₹1,25,000.50 ಇದೆ."
tokentax failure-lab
pytest -q
```

The Streamlit workbench contains a tokenizer receipt and heatmap, audit,
evaluation frontier, improvement ledger, and failure lab. Results appear only
after a corresponding workflow writes a local artifact; missing credentials,
data, model support, or results are shown as blocked/not measured rather than
replaced with illustrative values.

## Current run status

The public Belebele data downloader and schema checks are runnable. FLORES
requires the user's accepted gated access and local `HF_TOKEN`; OpenAI-
compatible evaluation requires its local API key, model/rate configuration,
and explicit paid consent. The local translation workbench and local model
evaluation need model downloads. Audit, quality, router, and adaptation
findings must not be claimed until their respective commands complete and
produce reports.

## Data and licensing

Downloaded datasets and generated experiment artifacts are local-only and
ignored by Git. Respect the FLORES evaluation-only terms and the source
licenses and attribution requirements of Wikimedia-derived training text.
