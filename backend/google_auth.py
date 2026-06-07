"""
google_auth.py — OAuth2 token management for the backend

Loads a refresh token saved by oauth_setup.py and returns authenticated
Google API clients. Auto-refreshes the access token when expired.
"""

import json
import os
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

TOKEN_FILE = Path(__file__).parent / "oauth_token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _load_token() -> Optional[dict]:
    env = os.environ.get("GOOGLE_OAUTH_TOKEN")
    if env:
        try:
            return json.loads(env)
        except json.JSONDecodeError as e:
            print(f"[ERROR] GOOGLE_OAUTH_TOKEN is not valid JSON: {e}")
            return None
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {TOKEN_FILE}: {e}")
            return None
    return None


def get_credentials() -> Optional[Credentials]:
    token_data = _load_token()
    if not token_data:
        return None
    try:
        creds = Credentials.from_authorized_user_info(token_data, scopes=SCOPES)
    except Exception as e:
        print(f"[ERROR] Failed to construct OAuth2 credentials: {e}")
        return None
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_data["token"] = creds.token
            if not os.environ.get("GOOGLE_OAUTH_TOKEN") and TOKEN_FILE.exists():
                with open(TOKEN_FILE, "w") as f:
                    json.dump(token_data, f, indent=2)
            print("[OK] Refreshed Google OAuth2 access token.")
        except Exception as e:
            print(f"[ERROR] Failed to refresh OAuth2 token: {e}")
            return None
    if not creds.valid:
        print("[ERROR] OAuth2 credentials are not valid.")
        return None
    return creds


def get_calendar_service_oauth():
    creds = get_credentials()
    if not creds:
        return None
    try:
        return build("calendar", "v3", credentials=creds)
    except Exception as e:
        print(f"[ERROR] Failed to build Calendar client: {e}")
        return None
