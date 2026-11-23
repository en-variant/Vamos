import re
import unicodedata


class SafetyGate:
    """Reject rewrites that alter protected facts or fail their recorded round-trip."""

    def __init__(self, min_backtranslation_chrf: float = 0.5):
        if not 0 <= min_backtranslation_chrf <= 1:
            raise ValueError("min_backtranslation_chrf must be between 0 and 1")
        self.min_backtranslation_chrf = min_backtranslation_chrf

    @staticmethod
    def _digits(text: str) -> list[str]:
        return re.findall(r"\d+(?:[.,]\d+)*", unicodedata.normalize("NFC", text))

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFC", text).split())

    def _backtranslation_chrf(self, original: str, round_trip: str) -> float:
        from sacrebleu.metrics import CHRF
        return CHRF(word_order=0).sentence_score(
            self._normalize(round_trip), [self._normalize(original)]
        ).score / 100

    def check(self, native: str, rewritten: str, metadata: dict | None = None) -> tuple[bool, str]:
        metadata = metadata or {}
        if metadata.get("fallback"):
            return False, metadata.get("reason", "rewrite strategy unavailable")
        native_digits = self._digits(native)
        rewritten_digits = self._digits(rewritten)
        if native_digits != rewritten_digits:
            return False, "digit sequence changed"
        names_native = set(re.findall(
            r"\b[A-Z][A-Za-z0-9._-]+(?:\s+[A-Z][A-Za-z0-9._-]+)*\b", native
        ))
        names_native.update(metadata.get("named_entities", []))
        names_rewritten = set(re.findall(
            r"\b[A-Z][A-Za-z0-9._-]+(?:\s+[A-Z][A-Za-z0-9._-]+)*\b", rewritten
        ))
        names_rewritten.update(metadata.get("named_entities_in_rewrite", []))
        if names_native and not names_native.issubset(names_rewritten):
            return False, "named entity not preserved"
        round_trip = metadata.get("round_trip")
        if metadata.get("round_trip_kind") == "transliteration":
            if not round_trip or self._normalize(native) != self._normalize(round_trip):
                return False, "romanization round-trip changed the source"
        elif metadata.get("round_trip_kind") == "translation":
            if not round_trip:
                return False, "back-translation missing"
            if self._digits(native) != self._digits(round_trip):
                return False, "back-translation changed digit sequence"
            if not names_native.issubset(set(re.findall(
                r"\b[A-Z][A-Za-z0-9._-]+(?:\s+[A-Z][A-Za-z0-9._-]+)*\b", round_trip
            )) | set(metadata.get("named_entities_in_round_trip", []))):
                return False, "back-translation did not preserve named entities"
            try:
                score = self._backtranslation_chrf(native, round_trip)
            except ImportError as exc:
                return False, f"cannot score back-translation: {exc}"
            if score < self.min_backtranslation_chrf:
                return False, f"back-translation chrF {score:.3f} below threshold"
        return True, "accepted"
