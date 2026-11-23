def find_break_even(
    native_latency_ms: float,
    translation_latency_ms: float,
    prompt_threshold_tokens: float,
    saved_fraction: float,
) -> dict:
    """Estimate the prompt length where measured token savings offset translation latency."""
    if native_latency_ms < 0 or translation_latency_ms < 0:
        raise ValueError("latencies cannot be negative")
    if prompt_threshold_tokens <= 0:
        raise ValueError("prompt_threshold_tokens must be positive")
    if not 0 <= saved_fraction <= 1:
        raise ValueError("saved_fraction must be between 0 and 1")
    if saved_fraction == 0:
        return {
            "status": "no_token_savings",
            "break_even_prompt_tokens": None,
            "native_latency_ms": native_latency_ms,
            "translation_latency_ms": translation_latency_ms,
        }
    ms_per_native_token = native_latency_ms / prompt_threshold_tokens
    if ms_per_native_token == 0:
        return {
            "status": "no_measured_native_latency",
            "break_even_prompt_tokens": None,
            "native_latency_ms": native_latency_ms,
            "translation_latency_ms": translation_latency_ms,
        }
    return {
        "status": "estimated_from_measured_latency",
        "model": "linear token-processing assumption",
        "native_latency_ms": native_latency_ms,
        "translation_latency_ms": translation_latency_ms,
        "native_ms_per_token": ms_per_native_token,
        "break_even_prompt_tokens": round(
            translation_latency_ms / (ms_per_native_token * saved_fraction), 2
        ),
        "saved_fraction": saved_fraction,
        "assumption": "processing latency scales linearly with input token count; generation overhead is not separated",
    }
