import json

import pytest

from tokentax_lab.audit.flores import download_belebele, load_local_flores


def test_local_flores_requires_parallel_1012_lines(tmp_path):
    for language in ("kn", "hi", "ta", "en"):
        (tmp_path / f"{language}.devtest").write_text(
            "\n".join(f"{language} {index}" for index in range(1012)),
            encoding="utf-8",
        )
    loaded = load_local_flores(tmp_path, ("kn", "hi", "ta", "en"))
    assert {len(rows) for rows in loaded.values()} == {1012}
    (tmp_path / "ta.devtest").write_text("only one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parallel|1,012"):
        load_local_flores(tmp_path, ("kn", "hi", "ta", "en"))


def test_belebele_downloader_validates_jsonl_and_writes_provenance(tmp_path, monkeypatch):
    row = {
        "link": "passage-1",
        "question_number": 1,
        "flores_passage": "ಪಠ್ಯ",
        "question": "ಪ್ರಶ್ನೆ?",
        "mc_answer1": "A1",
        "mc_answer2": "A2",
        "mc_answer3": "A3",
        "mc_answer4": "A4",
        "correct_answer_num": "1",
    }

    class Response:
        content = (json.dumps(row, ensure_ascii=False) + "\n").encode()
        text = content.decode()

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    loaded = download_belebele(
        tmp_path, languages={"kn": "kan_Knda", "en": "eng_Latn"}
    )
    assert loaded["kn"][0]["question_number"] == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["questions_per_language"] == {"kn": 1, "en": 1}


def test_belebele_downloader_reports_http_errors(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            raise RuntimeError("HTTP 403")

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="403"):
        download_belebele(tmp_path, languages={"kn": "kan_Knda"})
