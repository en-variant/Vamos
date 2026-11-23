from .optimizer.safety import SafetyGate


def examples() -> list[dict]:
    return [
        {"name":"Kanglish code-mixing","native":"ನಾನು Bangaloreಗೆ bus ಏರಿ ಹೋಗುತ್ತೇನೆ.", "breaking":"Romanization may lose script-boundary meaning"},
        {"name":"Proper-noun romanization","native":"ಕರ್ನಾಟಕದ Mysuru ನಲ್ಲಿದೆ.", "breaking":"Round-trip may alter a named entity"},
        {"name":"Digits and decimals","native":"ಬೆಲೆ ₹1,25,000.50 ಮತ್ತು 2026.", "breaking":"Digit separators or currency formats may change"},
        {"name":"Unicode zero width","native":"ಕನ್ನಡ\u200bಪಠ್ಯ", "breaking":"Cleanup removes an invisible joiner/context signal"},
        {"name":"Pivot ambiguity","native":"ಅವಳು mango ತಂದು ಬಿಟ್ಟಳು.", "breaking":"English loanword and pronoun meaning may not preserve"},
    ]


def run_example(example: dict, rewritten: str) -> dict:
    accepted, reason = SafetyGate().check(example["native"], rewritten, {})
    return {**example, "rewritten": rewritten, "accepted": accepted, "reason": reason}
