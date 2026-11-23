from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime, timezone
import json


def load_local_flores(directory: str | Path, languages: tuple[str, ...]) -> dict[str, list[str]]:
    """Load one .txt file per language; files must contain devtest sentences in parallel order."""
    directory = Path(directory)
    result = {}
    for lang in languages:
        candidates = [directory / f"{lang}.devtest", directory / f"{lang}.txt", directory / f"devtest.{lang}"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing FLORES devtest file for {lang}; tried: {', '.join(str(p) for p in candidates)}")
        result[lang] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lengths = {len(v) for v in result.values()}
    if len(lengths) != 1:
        raise ValueError(f"FLORES files are not parallel; lengths={lengths}")
    sentence_counts = {len(v) for v in result.values()}
    if sentence_counts != {1012}:
        raise ValueError(f"Expected 1,012 FLORES devtest sentences per language, got {sentence_counts}")
    return result


def load_from_huggingface(
    directory: str | Path = "data/flores",
    languages: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Download gated FLORES-200 devtest after access terms have been accepted."""
    languages = languages or {
        "kn": "kan_Knda",
        "hi": "hin_Deva",
        "ta": "tam_Taml",
        "en": "eng_Latn",
    }
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "FLORES-200 is gated. Accept its Hugging Face terms and set HF_TOKEN locally, "
            "or provide the four licensed local devtest files."
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before downloading FLORES.") from exc

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = {}
    for language, config in languages.items():
        dataset = load_dataset("facebook/flores", config, split="devtest", token=token)
        if "sentence" not in dataset.column_names:
            raise ValueError(f"Unexpected FLORES columns for {config}: {dataset.column_names}")
        sentences = [str(row["sentence"]) for row in dataset]
        if len(sentences) != 1012:
            raise ValueError(f"Expected 1,012 {config} devtest sentences, received {len(sentences)}")
        (directory / f"{language}.devtest").write_text(
            "\n".join(sentences) + "\n", encoding="utf-8"
        )
        output[language] = sentences
    if len({len(items) for items in output.values()}) != 1:
        raise ValueError("Downloaded FLORES language files do not contain matching sentence counts")
    (directory / "manifest.json").write_text(
        json.dumps({
            "dataset": "facebook/flores",
            "split": "devtest",
            "language_configs": languages,
            "sentences_per_language": 1012,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "gated_access_terms_accepted_by_user": True,
            "usage_note": "Evaluation only; do not use for model training or redistribute publicly.",
            "source_card": "https://huggingface.co/datasets/facebook/flores",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def download_belebele(
    directory: str | Path = "data/belebele",
    languages: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Fetch the public Belebele JSONL files from the dataset's official Hub repo."""
    import json
    import requests

    languages = languages or {
        "kn": "kan_Knda",
        "hi": "hin_Deva",
        "ta": "tam_Taml",
        "en": "eng_Latn",
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = {}
    for language, config in languages.items():
        path = directory / f"{language}.jsonl"
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            url = (
                "https://huggingface.co/datasets/facebook/belebele/resolve/"
                f"7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/{config}.jsonl"
            )
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            lines = response.text.splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        required = {
            "link", "question_number", "flores_passage", "question",
            "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
            "correct_answer_num",
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Unexpected Belebele schema in {path}")
        output[language] = rows
    (directory / "manifest.json").write_text(
        json.dumps({
            "dataset": "facebook/belebele",
            "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
            "split": "test",
            "language_configs": languages,
            "questions_per_language": {key: len(rows) for key, rows in output.items()},
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_card": "https://huggingface.co/datasets/facebook/belebele",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
