import os
import sys
import json

sys.path.append(os.path.abspath('.'))

from app import get_calendar_service, parse_slot_time, get_config
import datetime

service = get_calendar_service()
config = get_config()
calendar_id = config.get("google_calendar_id")

if service and calendar_id:
    try:
        slot_start = parse_slot_time("2026-06-08", "02:00 PM")
        slot_end = slot_start + datetime.timedelta(hours=1)
        event_body = {
            'summary': 'Interview with Test Candidate',
            'start': {
                'dateTime': slot_start.isoformat(),
                'timeZone': 'Asia/Kolkata',
            },
            'end': {
                'dateTime': slot_end.isoformat(),
                'timeZone': 'Asia/Kolkata',
            }
        }
        print("Inserting on shared calendar with no attendees and no conference data...")
        event = service.events().insert(
            calendarId=calendar_id,
            body=event_body
        ).execute()
        print("Success!")
        print("Event HTML Link:", event.get('htmlLink'))
    except Exception as e:
        print("Failed:", e)
else:
    print("Missing service or calendar_id")
