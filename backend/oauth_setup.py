"""
oauth_setup.py — One-time OAuth2 setup for Google Calendar (user authentication)

Run this once on a machine with a browser to authorize the app to act on your
Google account. It saves a refresh token that the backend uses to create events
with attendees and Google Meet links.

Prerequisites:
    1. Go to https://console.cloud.google.com/apis/credentials
    2. Create an OAuth 2.0 Client ID of type "Desktop app"
    3. Download the JSON and save it as backend/oauth_client.json
    4. Enable the Google Calendar API for your project
    5. pip install google-auth-oauthlib

Run:
    python oauth_setup.py
"""

import json
import os
import sys
from pathlib import Path

CLIENT_FILE = Path(__file__).parent / "oauth_client.json"
TOKEN_FILE = Path(__file__).parent / "oauth_token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def main():
    if not CLIENT_FILE.exists():
        print(f"[ERROR] Missing {CLIENT_FILE}")
        print("Download your OAuth2 client credentials from Google Cloud Console")
        print("and save them as backend/oauth_client.json")
        print()
        print("Steps:")
        print("  1. https://console.cloud.google.com/apis/credentials")
        print("  2. Create Credentials → OAuth client ID → Desktop app")
        print("  3. Download JSON → save as backend/oauth_client.json")
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("[ERROR] Missing dependency: google-auth-oauthlib")
        print("Install with: pip install google-auth-oauthlib")
        sys.exit(1)

    print("[INFO] Starting OAuth2 flow...")
    print(f"[INFO] Scopes: {SCOPES}")
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES)
    creds = flow.run_local_server(port=8765, open_browser=True)

    token_data = {
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
        "token": creds.token,
        "token_uri": creds.token_uri,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "type": "authorized_user",
    }

    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)

    print(f"\n[OK] Token saved to {TOKEN_FILE}")
    print("[OK] You can now run the backend. It will auto-refresh this token.")
    print()
    print("Add this to your deployment:")
    print("  - Set GOOGLE_OAUTH_TOKEN env var to the file contents, OR")
    print(f"  - Mount {TOKEN_FILE} at the same path in your container")


if __name__ == "__main__":
    main()
