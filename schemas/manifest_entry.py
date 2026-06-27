from dataclasses import dataclass, asdict
import json


VALID_SPLITS = frozenset({"train", "dev", "test"})
VALID_GENDERS = frozenset({"M", "F", "U"})
VALID_AGE_RANGES = frozenset({"18-30", "31-45", "46-60", "60+"})


@dataclass
class ManifestEntry:
    """Single record in master_manifest.jsonl. Schema is frozen — add fields with migration."""

    # Content
    audio_filepath: str
    text: str
    duration: float         # seconds, rounded to 3 dp

    # Language & speaker metadata
    language: str           # ISO 639-3: "yo" | "efi" | "ibb"
    speaker_id: str         # opaque string, stable per speaker across sessions
    speaker_gender: str     # "M" | "F" | "U" (unknown)
    speaker_age_range: str  # "18-30" | "31-45" | "46-60" | "60+"
    dialect: str            # free-text dialect label, "" if unknown

    # Technical
    sample_rate: int        # always 16000
    hash_id: str            # SHA-256[:16] of raw audio bytes (content-addressed)

    # Dataset split — must be speaker-disjoint (same speaker_id never in two splits)
    split: str              # "train" | "dev" | "test"

    # Quality flags — populated by audio_pipeline, used to filter training sets
    quality_snr_db: float   # estimated SNR; <15 dB is typically unusable
    quality_clipping: bool  # True if peak amplitude >= 0.99

    # Provenance
    created_at: str         # ISO 8601 UTC, e.g. "2026-06-27T00:00:00+00:00"

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "ManifestEntry":
        return cls(**json.loads(line.strip()))

    def is_clean(self, min_snr_db: float = 15.0) -> bool:
        """Returns True when this entry passes basic quality thresholds."""
        return not self.quality_clipping and self.quality_snr_db >= min_snr_db

    def validate_metadata(self) -> list[str]:
        errors = []
        if self.split not in VALID_SPLITS:
            errors.append(f"split must be one of {VALID_SPLITS}, got {self.split!r}")
        if self.speaker_gender not in VALID_GENDERS:
            errors.append(f"speaker_gender must be one of {VALID_GENDERS}, got {self.speaker_gender!r}")
        if self.speaker_age_range not in VALID_AGE_RANGES:
            errors.append(f"speaker_age_range must be one of {VALID_AGE_RANGES}, got {self.speaker_age_range!r}")
        if self.sample_rate != 16000:
            errors.append(f"sample_rate must be 16000, got {self.sample_rate}")
        if self.duration <= 0:
            errors.append("duration must be positive")
        return errors
