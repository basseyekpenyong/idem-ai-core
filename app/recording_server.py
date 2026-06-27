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
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from engine.audio_pipeline import manifest_stats, process_audio
from engine.script_generator import mock_scripts
from engine.validator import validate

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _REPO_ROOT / "master_manifest.jsonl"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "processed"

app = FastAPI(title="IdemAI Recording Studio", version="1.0.0")


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
  #status { color: #56ccf2; margin-top: 10px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 10px 0; }
  label { font-size: 13px; color: #aaa; }
</style>
</head>
<body>
<h1>🎙️ IdemAI Recording Studio</h1>

<div class="row">
  <label>Language</label>
  <select id="lang">
    <option value="yo">Yoruba</option>
    <option value="efi">Efik</option>
    <option value="ibb">Ibibio</option>
  </select>
  <label>Speaker ID</label>
  <input id="speaker-id" placeholder="e.g. spk_001" style="width:120px">
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
<p id="status"></p>

<script>
let mediaRecorder, audioBlob, chunks = [], recording = false;
const status = id => document.getElementById('status').textContent = id;

async function loadScript() {
  const lang = document.getElementById('lang').value;
  const res = await fetch('/scripts/' + lang + '?count=1');
  const data = await res.json();
  document.getElementById('script-box').textContent = data[0]?.text || 'No script available.';
  document.getElementById('rec-btn').disabled = false;
  document.getElementById('submit-btn').disabled = true;
  audioBlob = null;
  status('Script loaded. Press Record when ready.');
}

async function toggleRecord() {
  if (!recording) {
    chunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      audioBlob = new Blob(chunks, { type: 'audio/webm' });
      document.getElementById('submit-btn').disabled = false;
      status('Recording complete. Review and Submit.');
    };
    mediaRecorder.start();
    recording = true;
    document.getElementById('rec-btn').textContent = '⏹ Stop';
    status('Recording… speak clearly.');
  } else {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    recording = false;
    document.getElementById('rec-btn').textContent = '⏺ Record';
  }
}

async function submitRecording() {
  if (!audioBlob) return;
  const text = document.getElementById('script-box').textContent;
  const lang = document.getElementById('lang').value;
  const spk = document.getElementById('speaker-id').value || 'anon';
  const gender = document.getElementById('gender').value;
  const age = document.getElementById('age').value;

  const form = new FormData();
  form.append('audio', audioBlob, 'recording.webm');
  form.append('transcription', text);
  form.append('language_code', lang);
  form.append('speaker_id', spk);
  form.append('speaker_gender', gender);
  form.append('speaker_age_range', age);

  status('Submitting…');
  const res = await fetch('/submit', { method: 'POST', body: form });
  const data = await res.json();

  if (res.ok) {
    const clean = data.is_clean ? '✅ Clean' : '⚠️ Low quality';
    status(`Saved! ${clean} | Duration: ${data.duration}s | SNR: ${data.quality_snr_db} dB`);
    document.getElementById('submit-btn').disabled = true;
  } else {
    status('Error: ' + JSON.stringify(data.detail));
  }
}
</script>
</body>
</html>"""


@app.get("/studio", response_class=HTMLResponse)
def studio() -> HTMLResponse:
    """Serve the browser-based recording studio."""
    return HTMLResponse(_STUDIO_HTML)
