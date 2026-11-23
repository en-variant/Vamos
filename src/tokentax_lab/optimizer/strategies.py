from __future__ import annotations

import re
import unicodedata
import time


class Strategy:
    name = "strategy"

    def transform(self, text: str, language: str = "kn") -> tuple[str, dict]:
        return text, {"added_latency_ms": 0.0, "quality_risk": "low"}

    def postprocess_response(self, response: str, metadata: dict | None = None) -> str:
        return response


class NativeStrategy(Strategy):
    name = "native"


class UnicodeCleanupStrategy(Strategy):
    name = "unicode_cleanup"

    def transform(self, text: str, language: str = "kn") -> tuple[str, dict]:
        start = time.perf_counter()
        cleaned = unicodedata.normalize("NFC", text)
        cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned).strip()
        return cleaned, {
            "added_latency_ms": (time.perf_counter() - start) * 1000,
            "quality_risk": "very_low",
        }


class MixedLanguageStrategy(Strategy):
    name = "mixed_language"

    def transform(self, text: str, language: str = "kn") -> tuple[str, dict]:
        if language == "en":
            return text, {"added_latency_ms": 0.0, "quality_risk": "very_low"}
        language_name = {"kn": "Kannada", "hi": "Hindi", "ta": "Tamil"}.get(language, language)
        compiled = (
            f"Answer in English. Preserve the meaning of the {language_name} content.\n"
            f"<{language_name} content>\n{text}\n</{language_name} content>"
        )
        return compiled, {"added_latency_ms": 0.0, "quality_risk": "low"}


class RomanizeStrategy(Strategy):
    name = "romanize"
    SCRIPT_NAMES = {"kn": "KANNADA", "hi": "DEVANAGARI", "ta": "TAMIL"}
    SCHEME_NAMES = {"itrans": "ITRANS", "iso": "ISO", "casual": "ITRANS"}

    def __init__(self, scheme: str = "itrans"):
        if scheme.lower() not in self.SCHEME_NAMES:
            raise ValueError(f"Unsupported romanization scheme: {scheme}")
        self.scheme = scheme.lower()

    def transform(self, text: str, language: str = "kn") -> tuple[str, dict]:
        start = time.perf_counter()
        try:
            from indic_transliteration import sanscript
            if language not in self.SCRIPT_NAMES:
                raise ValueError(f"Unsupported language: {language}")
            source = getattr(sanscript, self.SCRIPT_NAMES[language])
            target = getattr(sanscript, self.SCHEME_NAMES[self.scheme])
            converted = sanscript.transliterate(text, source, target)
            if self.scheme == "casual":
                converted = converted.replace("A", "aa").replace("I", "ee").replace("U", "oo")
                round_trip_input = converted.replace("aa", "A").replace("ee", "I").replace("oo", "U")
            else:
                round_trip_input = converted
            round_trip = sanscript.transliterate(round_trip_input, target, source)
        except (ImportError, AttributeError, ValueError) as exc:
            return text, {
                "scheme": self.scheme,
                "fallback": True,
                "reason": f"Romanization unavailable: {exc}",
                "added_latency_ms": (time.perf_counter() - start) * 1000,
                "quality_risk": "high",
            }
        return converted, {
            "scheme": self.scheme,
            "round_trip": round_trip,
            "round_trip_kind": "transliteration",
            "added_latency_ms": (time.perf_counter() - start) * 1000,
            "quality_risk": "medium" if self.scheme != "casual" else "high",
        }


class PivotStrategy(Strategy):
    name = "pivot"
    LANGUAGE_CODES = {"kn": "kan_Knda", "hi": "hin_Deva", "ta": "tam_Taml", "en": "eng_Latn"}

    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(self.device)

    def _translate(self, text: str, source: str, target: str) -> str:
        self._tokenizer.src_lang = self.LANGUAGE_CODES[source]
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        target_id = self._tokenizer.convert_tokens_to_ids(self.LANGUAGE_CODES[target])
        output = self._model.generate(**inputs, forced_bos_token_id=target_id, max_new_tokens=256)
        return self._tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    def transform(self, text: str, language: str = "kn") -> tuple[str, dict]:
        start = time.perf_counter()
        try:
            if language not in self.LANGUAGE_CODES:
                raise ValueError(f"Unsupported pivot language: {language}")
            self._load()
            translated = self._translate(text, language, "en")
            round_trip = self._translate(translated, "en", language)
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            return text, {
                "added_latency_ms": (time.perf_counter() - start) * 1000,
                "quality_risk": "high",
                "fallback": True,
                "reason": f"Pivot unavailable: {exc}",
            }
        return translated, {
            "added_latency_ms": (time.perf_counter() - start) * 1000,
            "quality_risk": "medium",
            "source_language": language,
            "round_trip": round_trip,
            "round_trip_kind": "translation",
        }

    def postprocess_response(self, response: str, metadata: dict | None = None) -> str:
        metadata = metadata or {}
        if response.strip() in {"A", "B", "C", "D", "1", "2", "3", "4"}:
            return response.strip()
        language = metadata.get("source_language")
        if not language or language == "en":
            return response
        return self._translate(response, "en", language)
