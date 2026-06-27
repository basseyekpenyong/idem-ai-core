"""
Language profile registry for Efik, Ibibio, and Yoruba.

Design:
- All text must be NFC-normalised before validation (unicodedata.normalize("NFC", text)).
- After NFC, most precomposed characters (e.g. ẹ U+1EB9) are single codepoints.
- A small set of combining marks (tone marks) may still follow a base char when no
  precomposed form exists in Unicode (e.g. ẹ̀ = U+1EB9 + U+0300). These are captured in
  `extra_combining` and validated separately by checking Unicode category "Mn".
- To add a new language: define a LanguageProfile and register it in REGISTRY.
"""

import unicodedata
from dataclasses import dataclass, field
from typing import FrozenSet


# Combining marks shared across all three languages
_TONE_MARKS: FrozenSet[str] = frozenset({
    "̀",  # combining grave accent  (low tone)
    "́",  # combining acute accent  (high tone)
    "̂",  # combining circumflex    (falling/rising in some orthographies)
    "̄",  # combining macron        (mid tone)
    "̇",  # combining dot above     (used in some Yoruba orthographies)
    "̣",  # combining dot below     (NFD decomposition residue)
})

_SHARED_PUNCTUATION: FrozenSet[str] = frozenset(
    " .,!?;:'-–—\n\t"  # includes en-dash and em-dash
)


@dataclass(frozen=True)
class LanguageProfile:
    code: str       # ISO 639-3
    name: str
    base_chars: FrozenSet[str]
    extra_combining: FrozenSet[str] = field(default_factory=lambda: _TONE_MARKS)
    punctuation: FrozenSet[str] = field(default_factory=lambda: _SHARED_PUNCTUATION)

    def is_char_allowed(self, char: str) -> bool:
        cat = unicodedata.category(char)
        if cat == "Mn":  # Non-spacing combining mark
            return char in self.extra_combining
        return char in self.base_chars or char in self.punctuation

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFC", text).strip()


# ---------------------------------------------------------------------------
# Yoruba (yo)
# Alphabet: a b d e ẹ f g gb h i j k l m n o ọ p r s ṣ t u w y
# Tone marks applied via combining chars on base vowels and ẹ/ọ
# ---------------------------------------------------------------------------
_YORUBA_BASE: FrozenSet[str] = frozenset(
    # Standard Latin subset used in Yoruba
    "abdefghijklmnoprstuwygbABDEFGHIJKLMNOPRSTUWYGB"
    # Yoruba-specific precomposed characters (NFC)
    "ẹọṣ"       # e-dot-below, o-dot-below, s-dot-below
    "ẸỌṢ"       # uppercase
    # Precomposed tone-marked vowels that have Unicode precomposed forms
    "àáèéìíòóùú"
    "ÀÁÈÉÌÍÒÓÙÚ"
    # Precomposed ẹ and ọ with a single tone mark do NOT have precomposed Unicode
    # forms — they appear as ẹ + combining mark, handled by extra_combining
)

YORUBA = LanguageProfile(
    code="yo",
    name="Yoruba",
    base_chars=_YORUBA_BASE,
)


# ---------------------------------------------------------------------------
# Efik (efi)
# Closely related to Ibibio. Uses open-mid vowels ɔ/ɛ and nasal ŋ/ñ.
# ---------------------------------------------------------------------------
_EFIK_BASE: FrozenSet[str] = frozenset(
    "abdefghijklmnoprstuwABDEFGHIJKLMNOPRSTUW"
    "ɔɛñŋ"      # open-o, open-e, n-tilde, eng
    "ƆƐÑŊ"
    # Precomposed tone-marked vowels
    "àáèéìíòóùú"
    "ÀÁÈÉÌÍÒÓÙÚ"
    # ɔ and ɛ with tone marks have no precomposed Unicode — handled by extra_combining
)

EFIK = LanguageProfile(
    code="efi",
    name="Efik",
    base_chars=_EFIK_BASE,
)


# ---------------------------------------------------------------------------
# Ibibio (ibb)
# Dialect continuum with Efik. Identical technical character requirements.
# ---------------------------------------------------------------------------
_IBIBIO_BASE: FrozenSet[str] = frozenset(
    "abdefghijklmnoprstuwABDEFGHIJKLMNOPRSTUW"
    "ɔɛŋ"
    "ƆƐŊ"
    "àáèéìíòóùú"
    "ÀÁÈÉÌÍÒÓÙÚ"
)

IBIBIO = LanguageProfile(
    code="ibb",
    name="Ibibio",
    base_chars=_IBIBIO_BASE,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, LanguageProfile] = {
    "yo":  YORUBA,
    "efi": EFIK,
    "ibb": IBIBIO,
}


def get_profile(language_code: str) -> LanguageProfile:
    try:
        return REGISTRY[language_code]
    except KeyError:
        available = list(REGISTRY)
        raise KeyError(
            f"Unknown language code {language_code!r}. Available: {available}"
        ) from None
