"""
Google Drive tools for the Aziz Agent.

Authentication: OAuth2 (personal Google account).
- First run: opens a browser window to authorize. Token saved to ~/.idem-ai/token.json.
- Subsequent runs: token loaded and auto-refreshed silently.

You need a Google Cloud project with the Drive API enabled and a
credentials.json file downloaded. Run `python setup_drive_auth.py` once
to walk through this and complete the first-time login.

Tool capabilities exposed to Aziz:
  list_drive_files    — search files by name, type, or folder
  download_drive_file — download a file from Drive to local disk
  upload_to_drive     — upload a local file to a Drive folder
  rename_drive_file   — rename a file on Drive
  move_drive_file     — move a file to a different Drive folder
  generate_and_upload_notebook — generate a Whisper fine-tuning Colab notebook
                                 and upload it to Drive, ready to open
"""

from __future__ import annotations

import io
import os
from pathlib import Path

_TOKEN_PATH = Path.home() / ".idem-ai" / "token.json"
_CREDENTIALS_PATH = Path.home() / ".idem-ai" / "credentials.json"
_SCOPES = ["https://www.googleapis.com/auth/drive"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_service():
    """Return an authenticated Drive API service object."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError(
            "Google API libraries not installed. Run:\n"
            "  pip install google-api-python-client google-auth-httplib2 "
            "google-auth-oauthlib"
        )

    creds = None
    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {_CREDENTIALS_PATH}.\n"
                    "Run `python setup_drive_auth.py` to complete setup."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(_CREDENTIALS_PATH), _SCOPES
            )
            creds = flow.run_local_server(port=0)

        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_PATH.write_text(creds.to_json())

    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def list_drive_files(
    query: str = "",
    folder_id: str = "",
    mime_type: str = "",
    limit: int = 20,
) -> dict:
    """List files on Drive matching the given criteria."""
    service = _get_service()

    parts = ["trashed = false"]
    if folder_id:
        parts.append(f"'{folder_id}' in parents")
    if mime_type:
        parts.append(f"mimeType = '{mime_type}'")
    if query:
        parts.append(f"name contains '{query}'")

    q = " and ".join(parts)
    result = service.files().list(
        q=q,
        pageSize=limit,
        fields="files(id, name, mimeType, size, modifiedTime, parents)",
    ).execute()

    files = result.get("files", [])
    return {"count": len(files), "files": files}


def download_drive_file(file_id: str, local_path: str) -> dict:
    """Download a Drive file to local_path. Returns file metadata."""
    from googleapiclient.http import MediaIoBaseDownload

    service = _get_service()
    meta = service.files().get(fileId=file_id, fields="name,size,mimeType").execute()

    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    with open(local, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return {
        "downloaded": str(local),
        "name": meta["name"],
        "size_bytes": int(meta.get("size", 0)),
        "mime_type": meta.get("mimeType"),
    }


def upload_to_drive(
    local_path: str,
    folder_id: str = "",
    drive_filename: str = "",
) -> dict:
    """Upload a local file to Drive. Returns the new file's ID and URL."""
    from googleapiclient.http import MediaFileUpload
    import mimetypes

    service = _get_service()
    local = Path(local_path)

    name = drive_filename or local.name
    mime = mimetypes.guess_type(str(local))[0] or "application/octet-stream"

    metadata: dict = {"name": name}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaFileUpload(str(local), mimetype=mime, resumable=True)
    file = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, name, webViewLink",
    ).execute()

    return {
        "file_id": file["id"],
        "name": file["name"],
        "url": file.get("webViewLink", ""),
    }


def rename_drive_file(file_id: str, new_name: str) -> dict:
    """Rename a file on Drive."""
    service = _get_service()
    updated = service.files().update(
        fileId=file_id,
        body={"name": new_name},
        fields="id, name",
    ).execute()
    return {"file_id": updated["id"], "new_name": updated["name"]}


def move_drive_file(file_id: str, destination_folder_id: str) -> dict:
    """Move a file to a different Drive folder."""
    service = _get_service()
    file = service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents", []))

    updated = service.files().update(
        fileId=file_id,
        addParents=destination_folder_id,
        removeParents=previous_parents,
        fields="id, name, parents",
    ).execute()
    return {
        "file_id": updated["id"],
        "name": updated["name"],
        "moved_to": destination_folder_id,
    }


def generate_and_upload_notebook(
    language_code: str,
    manifest_drive_path: str,
    output_folder_id: str = "",
    model_size: str = "small",
) -> dict:
    """Generate a Whisper fine-tuning Colab notebook and upload it to Drive."""
    from agent.tools.colab_generator import generate_whisper_notebook
    import tempfile

    notebook_json = generate_whisper_notebook(
        language_code=language_code,
        manifest_drive_path=manifest_drive_path,
        model_size=model_size,
    )

    notebook_name = f"idem_train_{language_code}_whisper_{model_size}.ipynb"
    with tempfile.NamedTemporaryFile(
        suffix=".ipynb", mode="w", encoding="utf-8", delete=False
    ) as tmp:
        import json
        json.dump(notebook_json, tmp, ensure_ascii=False, indent=1)
        tmp_path = tmp.name

    try:
        result = upload_to_drive(tmp_path, output_folder_id, notebook_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        **result,
        "instruction": (
            f"Notebook '{notebook_name}' uploaded to Drive. "
            "Open it in Google Colab and click Runtime > Run All."
        ),
    }


# ---------------------------------------------------------------------------
# Tool definitions (consumed by AzizOrchestrator)
# ---------------------------------------------------------------------------

DRIVE_TOOLS: list[dict] = [
    {
        "name": "list_drive_files",
        "description": (
            "List or search files on Google Drive. Use when the user asks to find, "
            "list, or search for files on their Drive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string", "description": "Filename search term"},
                "folder_id": {"type": "string", "description": "Drive folder ID to search in"},
                "mime_type": {"type": "string", "description": "Filter by MIME type, e.g. audio/wav"},
                "limit":     {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "download_drive_file",
        "description": (
            "Download a file from Google Drive to the local machine. "
            "Use when the user wants to pull audio, text, or any file from Drive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id":    {"type": "string", "description": "Google Drive file ID"},
                "local_path": {"type": "string", "description": "Local path to save the file to"},
            },
            "required": ["file_id", "local_path"],
        },
    },
    {
        "name": "upload_to_drive",
        "description": (
            "Upload a local file to Google Drive. Use when the user wants to save "
            "processed audio, the manifest, or a notebook to Drive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "local_path":    {"type": "string"},
                "folder_id":     {"type": "string", "description": "Destination Drive folder ID"},
                "drive_filename":{"type": "string", "description": "Name to give the file on Drive"},
            },
            "required": ["local_path"],
        },
    },
    {
        "name": "rename_drive_file",
        "description": "Rename a file on Google Drive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id":  {"type": "string"},
                "new_name": {"type": "string"},
            },
            "required": ["file_id", "new_name"],
        },
    },
    {
        "name": "move_drive_file",
        "description": "Move a file to a different folder on Google Drive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id":               {"type": "string"},
                "destination_folder_id": {"type": "string"},
            },
            "required": ["file_id", "destination_folder_id"],
        },
    },
    {
        "name": "generate_and_upload_notebook",
        "description": (
            "Generate a Whisper fine-tuning Colab notebook for a language and upload it "
            "to Drive. Use when the user wants to train a model, fine-tune Whisper, "
            "or start a training run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language_code":       {"type": "string", "enum": ["yo", "efi", "ibb"]},
                "manifest_drive_path": {"type": "string", "description": "Path to manifest.jsonl on Drive (e.g. /IdemAI/master_manifest.jsonl)"},
                "output_folder_id":    {"type": "string", "description": "Drive folder ID to save the notebook"},
                "model_size":          {"type": "string", "enum": ["tiny", "small", "medium"], "default": "small"},
            },
            "required": ["language_code", "manifest_drive_path"],
        },
    },
]


DRIVE_EXECUTOR: dict = {
    "list_drive_files":            lambda p: list_drive_files(**p),
    "download_drive_file":         lambda p: download_drive_file(**p),
    "upload_to_drive":             lambda p: upload_to_drive(**p),
    "rename_drive_file":           lambda p: rename_drive_file(**p),
    "move_drive_file":             lambda p: move_drive_file(**p),
    "generate_and_upload_notebook":lambda p: generate_and_upload_notebook(**p),
}
