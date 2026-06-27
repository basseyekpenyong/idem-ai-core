"""
Whisper fine-tuning Colab notebook generator.

Generates a complete, ready-to-run .ipynb file that:
  1. Mounts Google Drive
  2. Reads master_manifest.jsonl from Drive
  3. Fine-tunes openai/whisper-{size} using HuggingFace Transformers
  4. Evaluates on the dev split (Word Error Rate)
  5. Saves the model and processor back to Drive

Supports 1–8 hours of training data on Colab free T4 GPU:
  - whisper-tiny:   ~30 min for 1h data,  ~3h for 8h data
  - whisper-small:  ~1h  for 1h data,  ~6h for 8h data  ← recommended
  - whisper-medium: needs Colab Pro for 8h data

Language codes → Whisper language tokens:
  yo  (Yoruba)  → "yo"  (supported natively in Whisper)
  efi (Efik)    → None  (unsupported; train without forced language token)
  ibb (Ibibio)  → None  (unsupported; train without forced language token)
"""

from __future__ import annotations

_LANG_TO_WHISPER: dict[str, str | None] = {
    "yo":  "yo",    # Yoruba — supported in Whisper vocabulary
    "efi": None,    # Efik — not in Whisper; no forced token, model learns from data
    "ibb": None,    # Ibibio — not in Whisper; same approach
}

_LANG_NAMES: dict[str, str] = {
    "yo":  "Yoruba",
    "efi": "Efik",
    "ibb": "Ibibio",
}


def _cell(source: str, cell_type: str = "code") -> dict:
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source,
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": ""},
        "outputs": [],
        "source": source,
    }


def generate_whisper_notebook(
    language_code: str,
    manifest_drive_path: str,
    model_size: str = "small",
) -> dict:
    """
    Generate a Whisper fine-tuning Jupyter notebook as a dict (serialisable to .ipynb).

    Args:
        language_code:       "yo" | "efi" | "ibb"
        manifest_drive_path: Path to master_manifest.jsonl on Google Drive,
                             e.g. "/IdemAI/master_manifest.jsonl"
        model_size:          "tiny" | "small" | "medium"

    Returns:
        A dict representing the .ipynb notebook structure.
    """
    lang_name = _LANG_NAMES.get(language_code, language_code)
    whisper_lang = _LANG_TO_WHISPER.get(language_code)
    model_id = f"openai/whisper-{model_size}"
    forced_lang_line = (
        f'processor.tokenizer.set_prefix_tokens(language="{whisper_lang}", task="transcribe")'
        if whisper_lang
        else "# Efik/Ibibio not in Whisper vocab — training without forced language token"
    )
    forced_ids_line = (
        f'model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="{whisper_lang}", task="transcribe")'
        if whisper_lang
        else "model.config.forced_decoder_ids = None"
    )

    cells = [
        _cell(f"# IdemAI — Whisper Fine-tuning: {lang_name} (`{language_code}`)\n\n"
              f"Model: `{model_id}` | Data: `{manifest_drive_path}`\n\n"
              "**Runtime → Change runtime type → T4 GPU** before running.",
              "markdown"),

        _cell("""\
# ── Cell 1: Mount Google Drive ──────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
"""),

        _cell("""\
# ── Cell 2: Install dependencies ─────────────────────────────────────────────
!pip install -q transformers datasets evaluate jiwer accelerate soundfile
"""),

        _cell(f"""\
# ── Cell 3: Configuration ────────────────────────────────────────────────────
import os

LANGUAGE_CODE  = "{language_code}"
LANGUAGE_NAME  = "{lang_name}"
MODEL_ID       = "{model_id}"
MANIFEST_PATH  = "/content/drive/MyDrive{manifest_drive_path}"
OUTPUT_DIR     = f"/content/drive/MyDrive/IdemAI/models/whisper-{model_size}-{{LANGUAGE_CODE}}"
AUDIO_BASE_DIR = "/content/drive/MyDrive"   # adjust if audio files are elsewhere

# Training hyperparameters — tuned for 1-8h data on T4 (15GB VRAM)
TRAIN_BATCH_SIZE  = 16
EVAL_BATCH_SIZE   = 8
LEARNING_RATE     = 1e-5
WARMUP_STEPS      = 100
MAX_STEPS         = 4000    # increase to 8000 for 8h+ data
SAVE_STEPS        = 500
EVAL_STEPS        = 500
GRADIENT_ACCUM    = 2       # effective batch = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Training {{LANGUAGE_NAME}} ({LANGUAGE_CODE})")
print(f"Model:    {{MODEL_ID}}")
print(f"Output:   {{OUTPUT_DIR}}")
"""),

        _cell("""\
# ── Cell 4: Load manifest and build HuggingFace dataset ──────────────────────
import json
import datasets
from pathlib import Path

def load_manifest(path, language_code, audio_base):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["language"] != language_code:
                continue
            # Resolve audio path: try absolute first, then relative to Drive
            audio = entry["audio_filepath"]
            if not Path(audio).exists():
                audio = audio_base + "/" + audio.lstrip("/")
            records.append({
                "audio":  audio,
                "text":   entry["text"],
                "split":  entry["split"],
                "duration": entry["duration"],
                "clean":  not entry["quality_clipping"] and entry["quality_snr_db"] >= 15.0,
            })
    return records

all_records = load_manifest(MANIFEST_PATH, LANGUAGE_CODE, AUDIO_BASE_DIR)
train_records = [r for r in all_records if r["split"] == "train" and r["clean"]]
dev_records   = [r for r in all_records if r["split"] == "dev"   and r["clean"]]

print(f"Train: {len(train_records)} clean utterances")
print(f"Dev:   {len(dev_records)} clean utterances")
print(f"Total train hours: {sum(r['duration'] for r in train_records)/3600:.2f}h")

train_ds = datasets.Dataset.from_list(train_records)
dev_ds   = datasets.Dataset.from_list(dev_records)
"""),

        _cell("""\
# ── Cell 5: Load processor and model ─────────────────────────────────────────
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.use_cache = False

""" + forced_lang_line + "\n" + forced_ids_line + """

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = model.to(device)
"""),

        _cell("""\
# ── Cell 6: Feature extraction ────────────────────────────────────────────────
import soundfile as sf
import numpy as np

TARGET_SR = 16_000

def prepare_batch(batch):
    audio_arrays, texts = [], []
    for path, text in zip(batch["audio"], batch["text"]):
        try:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            # Resample if needed (files should already be 16kHz)
            if sr != TARGET_SR:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(TARGET_SR, sr)
                audio = resample_poly(audio, TARGET_SR // g, sr // g)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            audio = np.zeros(TARGET_SR, dtype=np.float32)
        audio_arrays.append(audio)
        texts.append(text)

    inputs = processor(
        audio_arrays,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="longest",
    )
    labels = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=448,
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

print("Preprocessing train set...")
train_ds = train_ds.map(prepare_batch, batched=True, batch_size=8,
                         remove_columns=train_ds.column_names)
print("Preprocessing dev set...")
dev_ds = dev_ds.map(prepare_batch, batched=True, batch_size=8,
                     remove_columns=dev_ds.column_names)
"""),

        _cell("""\
# ── Cell 7: Data collator ─────────────────────────────────────────────────────
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
"""),

        _cell("""\
# ── Cell 8: Evaluation metric (WER) ──────────────────────────────────────────
import evaluate

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
"""),

        _cell(f"""\
# ── Cell 9: Training ──────────────────────────────────────────────────────────
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Starting training...")
trainer.train()
"""),

        _cell("""\
# ── Cell 10: Save model and processor to Drive ────────────────────────────────
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")
print("Open Drive to find your model. Load it with:")
print(f"  from transformers import WhisperForConditionalGeneration, WhisperProcessor")
print(f"  model = WhisperForConditionalGeneration.from_pretrained('{OUTPUT_DIR}')")
print(f"  processor = WhisperProcessor.from_pretrained('{OUTPUT_DIR}')")
"""),

        _cell("""\
# ── Cell 11: Quick inference test ─────────────────────────────────────────────
# Test the trained model on the first 3 dev utterances
import soundfile as sf
import numpy as np

model.eval()
print("Sample predictions on dev set:\\n")

for i, sample in enumerate(dev_ds.select(range(min(3, len(dev_ds))))):
    input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    reference = processor.tokenizer.decode(
        [t for t in sample["labels"] if t != -100], skip_special_tokens=True
    )
    print(f"[{i+1}] REF: {reference}")
    print(f"     HYP: {transcription}")
    print()
"""),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
