"""
One-time Google Drive OAuth2 setup.

Run this once before using any Drive features:
    python setup_drive_auth.py

What this does:
  1. Checks that credentials.json exists in ~/.idem-ai/
  2. Opens a browser window to authorize access to your Google Drive
  3. Saves the access + refresh token to ~/.idem-ai/token.json
  4. All future Drive calls use the saved token (auto-refreshed silently)

How to get credentials.json (takes ~5 minutes):
  1. Go to https://console.cloud.google.com/
  2. Create a new project (or select an existing one)
  3. Enable the Google Drive API:
       APIs & Services > Library > search "Google Drive API" > Enable
  4. Create OAuth2 credentials:
       APIs & Services > Credentials > + Create Credentials > OAuth client ID
       Application type: Desktop app
       Name: idem-ai-local
       Click Create
  5. Download the JSON file and save it to:
       C:\\Users\\<you>\\.idem-ai\\credentials.json   (Windows)
       ~/.idem-ai/credentials.json                    (Mac/Linux)
  6. Run this script.
"""

import sys
from pathlib import Path

_CREDS = Path.home() / ".idem-ai" / "credentials.json"
_TOKEN = Path.home() / ".idem-ai" / "token.json"
_SCOPES = ["https://www.googleapis.com/auth/drive"]


def main() -> None:
    print("IdemAI — Google Drive Auth Setup")
    print("=" * 40)

    # Check dependencies
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
    except ImportError:
        print("\nMissing Google API libraries. Install them first:\n")
        print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib\n")
        sys.exit(1)

    # Check credentials file
    if not _CREDS.exists():
        print(f"\ncredentials.json not found at:\n  {_CREDS}\n")
        print("Follow the steps in the docstring at the top of this file to create one.")
        print("Google Cloud Console: https://console.cloud.google.com/")
        sys.exit(1)

    print(f"\nFound credentials: {_CREDS}")
    print("Opening browser for authorization...\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS), _SCOPES)
    creds = flow.run_local_server(port=0)

    _TOKEN.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN.write_text(creds.to_json())
    print(f"\nToken saved to: {_TOKEN}")

    # Quick verify
    service = build("drive", "v3", credentials=creds)
    about = service.about().get(fields="user").execute()
    user = about.get("user", {})
    print(f"\nAuthorized as: {user.get('displayName')} ({user.get('emailAddress')})")
    print("\nSetup complete. You can now use Drive commands via the Aziz Agent.")
    print("\nExample commands:")
    print('  aziz.run("list my Drive files")')
    print('  aziz.run("download the Yoruba audio folder from Drive")')
    print('  aziz.run("generate a Whisper training notebook for Yoruba and upload to Drive")')


if __name__ == "__main__":
    main()
