"""
Text validator — NFC normalization + language-profile whitelist check.

Usage:
    result = validate("Ẹ̀ ní bẹ", "yo")
    if not result.is_valid:
        print(result.errors)
    else:
        store(result.normalized_text)
"""

import unicodedata
from dataclasses import dataclass, field
from config.language_profiles import get_profile

MIN_WORDS = 2
MAX_WORDS = 60   # ~30 seconds at 2 words/sec — reject absurdly long lines


@dataclass
class ValidationResult:
    is_valid: bool
    normalized_text: str
    errors: list[str] = field(default_factory=list)


def validate(text: str, language_code: str) -> ValidationResult:
    """
    Validate and normalise `text` against the language profile for `language_code`.

    Steps:
    1. NFC-normalise (composing precomposed forms where Unicode provides them).
    2. Strip leading/trailing whitespace.
    3. Check character-by-character: base chars and permitted combining marks.
    4. Check word-count bounds.

    Returns a ValidationResult. Call is_valid to gate downstream processing.
    """
    profile = get_profile(language_code)
    normalized = profile.normalize(text)

    errors: list[str] = []

    if not normalized:
        return ValidationResult(False, normalized, ["Text is empty after normalization"])

    # --- Character check ---
    illegal: set[str] = set()
    for char in normalized:
        if not profile.is_char_allowed(char):
            illegal.add(f"U+{ord(char):04X} ({char!r})")

    if illegal:
        errors.append(
            f"Illegal characters for language {language_code!r}: "
            + ", ".join(sorted(illegal))
        )

    # --- Word count ---
    words = normalized.split()
    if len(words) < MIN_WORDS:
        errors.append(
            f"Too short: {len(words)} word(s), minimum is {MIN_WORDS}"
        )
    if len(words) > MAX_WORDS:
        errors.append(
            f"Too long: {len(words)} words, maximum is {MAX_WORDS} "
            f"(split into shorter sentences)"
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        normalized_text=normalized,
        errors=errors,
    )


def validate_batch(
    pairs: list[tuple[str, str]]
) -> list[ValidationResult]:
    """Validate a list of (text, language_code) pairs."""
    return [validate(text, lang) for text, lang in pairs]
