"""
Recording server — FastAPI backend for browser-based audio capture.

Uses the browser's native MediaRecorder API (via the embedded HTML page) to
capture audio at the correct quality, then accepts the blob via POST, processes
it through the audio pipeline, and appends to master_manifest.jsonl.

Why not Streamlit? Streamlit re-runs the entire script on each interaction, which
fights MediaRecorder's stateful lifecycle. FastAPI + plain HTML is far more reliable
for real-time audio capture.

Run:
    uvicorn app.recording_server:app --reload --port 8001

Then open: http://localhost:8001/studio
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from engine.audio_pipeline import manifest_stats, process_audio
from engine.script_generator import mock_scripts
from engine.validator import validate

# Store data OUTSIDE OneDrive to avoid sync-lock conflicts with soundfile
_DATA_ROOT        = Path.home() / "idem-ai-data"
_DEFAULT_MANIFEST = _DATA_ROOT / "master_manifest.jsonl"
_DEFAULT_OUTPUT   = _DATA_ROOT / "processed"

app = FastAPI(title="IdemAI Recording Studio", version="1.0.0")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/studio")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/scripts/{language_code}")
def get_scripts(language_code: str, count: int = 5) -> JSONResponse:
    """Return script chunks to record for the given language."""
    try:
        chunks = mock_scripts(language_code, count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse([
        {"text": c.text, "word_count": c.word_count, "in_target_range": c.in_target_range}
        for c in chunks
    ])


@app.post("/submit")
async def submit_recording(
    audio: UploadFile,
    transcription: Annotated[str, Form()],
    language_code: Annotated[str, Form()],
    speaker_id: Annotated[str, Form()],
    speaker_gender: Annotated[str, Form()] = "U",
    speaker_age_range: Annotated[str, Form()] = "18-30",
    dialect: Annotated[str, Form()] = "",
    split: Annotated[str, Form()] = "train",
) -> JSONResponse:
    """
    Validate transcription, process audio, and append to the manifest.
    Returns the manifest entry summary.
    """
    # 1. Validate the transcription first — fast, no disk I/O
    val = validate(transcription, language_code)
    if not val.is_valid:
        raise HTTPException(
            status_code=422,
            detail={"validation_errors": val.errors},
        )

    # 2. Write uploaded audio to a temp file (soundfile needs a real path)
    audio_bytes = await audio.read()
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        _DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
        entry = process_audio(
            audio_path=tmp_path,
            transcription=val.normalized_text,
            language_code=language_code,
            speaker_id=speaker_id,
            speaker_gender=speaker_gender,
            speaker_age_range=speaker_age_range,
            dialect=dialect,
            split=split,
            output_dir=_DEFAULT_OUTPUT,
            manifest_path=_DEFAULT_MANIFEST,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({
        "hash_id": entry.hash_id,
        "duration": entry.duration,
        "quality_snr_db": entry.quality_snr_db,
        "quality_clipping": entry.quality_clipping,
        "is_clean": entry.is_clean(),
        "audio_filepath": entry.audio_filepath,
    })


@app.get("/status")
def get_status() -> JSONResponse:
    return JSONResponse(manifest_stats(_DEFAULT_MANIFEST))


# ---------------------------------------------------------------------------
# Embedded recording studio HTML
# ---------------------------------------------------------------------------
_STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IdemAI Recording Studio</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; background: #0f1117; color: #fafafa; }
  h1 { color: #f2994a; }
  select, input, button { padding: 8px 14px; border-radius: 6px; border: 1px solid #444; background: #1e2130; color: #fafafa; font-size: 15px; }
  button { cursor: pointer; background: #f2994a; color: #000; border: none; font-weight: 600; }
  button:disabled { background: #555; color: #999; cursor: not-allowed; }
  #script-box { background: #1e2130; border-radius: 8px; padding: 20px; margin: 20px 0; font-size: 18px; line-height: 1.6; min-height: 60px; }
  #status { margin-top: 10px; min-height: 1.4em; }
  .status-ok  { color: #6fcf97; }
  .status-err { color: #eb5757; }
  .status-inf { color: #56ccf2; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 10px 0; }
  label { font-size: 13px; color: #aaa; }
</style>
</head>
<body>
<h1>🎙️ IdemAI Recording Studio</h1>

<div class="row">
  <label>Language</label>
  <select id="lang" onchange="onLangChange()">
    <option value="efi">Efik</option>
    <option value="ibb">Ibibio</option>
    <option value="en_NG">Nigerian English</option>
  </select>
  <label>Speaker ID</label>
  <input id="speaker-id" style="width:160px">
  <label>Gender</label>
  <select id="gender"><option value="M">M</option><option value="F">F</option><option value="U">U</option></select>
  <label>Age</label>
  <select id="age"><option>18-30</option><option>31-45</option><option>46-60</option><option>60+</option></select>
</div>

<div class="row">
  <button onclick="loadScript()">📋 Load Script</button>
  <button id="rec-btn" onclick="toggleRecord()" disabled>⏺ Record</button>
  <button id="submit-btn" onclick="submitRecording()" disabled>✅ Submit</button>
</div>

<div id="script-box">Press "Load Script" to get a sentence to record.</div>
<p id="status" class="status-inf"></p>

<script>
// ── WAV encoder (pure JS — no webm/ffmpeg needed) ───────────────────────────
function encodeWAV(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v   = new DataView(buf);
  function ws(o, s) { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); }
  ws(0,  'RIFF'); v.setUint32(4, 36 + samples.length * 2, true);
  ws(8,  'WAVE'); ws(12, 'fmt ');
  v.setUint32(16, 16, true);            // PCM chunk size
  v.setUint16(20,  1, true);            // PCM format
  v.setUint16(22,  1, true);            // mono
  v.setUint32(24, sampleRate,     true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);             // block align
  v.setUint16(34, 16, true);            // 16-bit
  ws(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0, o = 44; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// ── State ────────────────────────────────────────────────────────────────────
let audioCtx = null, source = null, processor = null, analyser = null;
let pcmBufs = [], audioBlob = null, recording = false;
let silenceTimer = null;
let speakerEdited = false;  // true once user manually changes the field
const SILENCE_THRESHOLD = 0.01;  // RMS below this = silence
const SILENCE_SECONDS   = 2.0;   // stop after 2s of silence

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status-' + (cls || 'inf');
}

const SPEAKER_PREFIX = { 'efi': 'efi', 'ibb': 'ibi', 'en_NG': 'eng' };
function autoSpeakerId(lang) {
  return (SPEAKER_PREFIX[lang] || lang) + '_speaker001';
}

function onLangChange() {
  if (!speakerEdited) {
    document.getElementById('speaker-id').value = autoSpeakerId(
      document.getElementById('lang').value
    );
  }
  // Reset recording state when language changes
  audioBlob = null;
  document.getElementById('submit-btn').disabled = true;
}

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const spkInput = document.getElementById('speaker-id');
  spkInput.value = autoSpeakerId(document.getElementById('lang').value);
  spkInput.addEventListener('input', () => { speakerEdited = true; });
});

// ── Script loading ────────────────────────────────────────────────────────────
async function loadScript() {
  const lang = document.getElementById('lang').value;
  try {
    const res  = await fetch('/scripts/' + lang + '?count=1');
    const data = await res.json();
    document.getElementById('script-box').textContent = data[0]?.text || 'No script available.';
    document.getElementById('rec-btn').disabled = false;
    document.getElementById('submit-btn').disabled = true;
    audioBlob = null;
    setStatus('Script loaded. Press Record when ready.');
  } catch (e) {
    setStatus('Could not load script: ' + e.message, 'err');
  }
}

// ── Recording ─────────────────────────────────────────────────────────────────
async function toggleRecord() {
  if (!recording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx  = new AudioContext();
      source    = audioCtx.createMediaStreamSource(stream);
      analyser  = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      pcmBufs   = [];

      processor.onaudioprocess = e => {
        const buf = new Float32Array(e.inputBuffer.getChannelData(0));
        pcmBufs.push(new Float32Array(buf));
        // RMS energy check for silence detection
        const rms = Math.sqrt(buf.reduce((s, v) => s + v*v, 0) / buf.length);
        if (rms < SILENCE_THRESHOLD) {
          if (!silenceTimer) {
            silenceTimer = setTimeout(() => { if (recording) toggleRecord(); }, SILENCE_SECONDS * 1000);
          }
        } else {
          clearTimeout(silenceTimer); silenceTimer = null;
        }
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      recording = true;
      document.getElementById('rec-btn').textContent = '⏹ Stop';
      document.getElementById('submit-btn').disabled = true;
      setStatus('🔴 Recording… speak clearly. Auto-stops after 2s of silence.');
    } catch (e) {
      setStatus('Microphone error: ' + e.message, 'err');
    }
  } else {
    // Stop and encode as WAV
    clearTimeout(silenceTimer); silenceTimer = null;
    processor.disconnect(); source.disconnect();
    const sr = audioCtx.sampleRate;
    audioCtx.close();
    const total = pcmBufs.reduce((s, b) => s + b.length, 0);
    const all   = new Float32Array(total);
    let off = 0;
    for (const b of pcmBufs) { all.set(b, off); off += b.length; }
    audioBlob = encodeWAV(all, sr);
    recording = false;
    document.getElementById('rec-btn').textContent = '⏺ Record';
    document.getElementById('submit-btn').disabled = false;
    setStatus('Recording complete (' + (total / sr).toFixed(1) + 's). Press Submit.');
  }
}

// ── Submit ────────────────────────────────────────────────────────────────────
async function submitRecording() {
  if (!audioBlob) return;
  const text   = document.getElementById('script-box').textContent;
  const lang   = document.getElementById('lang').value;
  const spk    = document.getElementById('speaker-id').value || autoSpeakerId(lang);
  const gender = document.getElementById('gender').value;
  const age    = document.getElementById('age').value;

  const form = new FormData();
  form.append('audio',            audioBlob, 'recording.wav');
  form.append('transcription',    text);
  form.append('language_code',    lang);
  form.append('speaker_id',       spk);
  form.append('speaker_gender',   gender);
  form.append('speaker_age_range', age);

  setStatus('Submitting…');
  document.getElementById('submit-btn').disabled = true;
  try {
    const res  = await fetch('/submit', { method: 'POST', body: form });
    const data = await res.json();
    if (res.ok) {
      const q = data.is_clean ? '✅ Clean' : '⚠️ Low quality';
      setStatus(q + ' | Duration: ' + data.duration + 's | SNR: ' + data.quality_snr_db + ' dB', 'ok');
    } else {
      setStatus('Rejected: ' + JSON.stringify(data.detail), 'err');
      document.getElementById('submit-btn').disabled = false;
    }
  } catch (e) {
    setStatus('Network error: ' + e.message, 'err');
    document.getElementById('submit-btn').disabled = false;
  }
}
</script>
</body>
</html>"""


@app.get("/studio", response_class=HTMLResponse)
def studio() -> HTMLResponse:
    """Serve the browser-based recording studio."""
    return HTMLResponse(_STUDIO_HTML)
