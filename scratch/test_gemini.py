import os
import sys
import json

# Add backend directory to path
sys.path.append('/Users/shaanraza/.gemini/antigravity/scratch/shaan-chatbot/backend')

from app import init_gemini, gemini_client, GEMINI_MODEL

print("Model:", GEMINI_MODEL)
print("GEMINI_API_KEY environment variable:", os.environ.get("GEMINI_API_KEY", "NOT SET")[:10] + "..." if os.environ.get("GEMINI_API_KEY") else "NOT SET")

success = init_gemini()
print("Client initialization success:", success)

if success and gemini_client:
    try:
        print("Calling generate_content...")
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Hello, reply with 'test ok' if you hear me."
        )
        print("Response received:")
        print(response.text)
    except Exception as e:
        print("Error calling Gemini:", e)
        import traceback
        traceback.print_exc()
else:
    print("Cannot test: Gemini client not initialized.")
